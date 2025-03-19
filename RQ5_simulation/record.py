import os
import bleu
import logging
import argparse
import jsonlines

import numpy as np

class SimulationRecord:
    def __init__(self, commit_url, args):
        self.commit_url = commit_url
        self.simulation_order = []
        self.locator_runtime = []
        self.generator_runtime = []
        self.locator_k_list = [1, 3, 5, 10, float('inf')]
        self.generator_k_list = [1, 3, 5, 10]
        
        if args and not args.label_correction:
            self.predicted_labels_correct = 0
            
        # Initialize match count lists as instance variables
        for k in self.locator_k_list:
            setattr(self, f"num_match_@{k}", [])
            setattr(self, f"num_allowed_match_@{k}", [])
        for k in self.generator_k_list:
            setattr(self, f"bleu_@{k}", [])
           
    def register_locator_match(self, predictions: list[dict], args: argparse.Namespace, logger: logging.Logger):
        if args.system == "TRACE-wo-Invoker":
            # For TRACE-wo-Invoker, their recommendation will direct apply without user confirmation
            # If exists any non-actionable location, we should not count it as a match
            LSP_false_positive_edit = False
            for pidx, prediction in enumerate(predictions, start=1):
                if not prediction["matched"] and prediction["lsp_service"] != "normal":
                    # if this prediction does not match with ground-truth, meanwhile it is recommended by lsp services
                    LSP_false_positive_edit = True
                    break
            if LSP_false_positive_edit: # if there exist such false positive ones, we have to reject the entire round of recommendations
                logger.info("Blindly invoking LSP has created false positives")
                predictions = []
            
        # Initialize temporary counters
        match_counts = {k: 0 for k in self.locator_k_list}
        matched_edit_idxs = []
        rank1_matched_edit_idxs = []
        service_for_matched_edit = {}
        
        # Count matches
        for prediction in predictions:
            if prediction["matched"]:
                if prediction["rank"] == 1:
                    rank1_matched_edit_idxs.append(prediction["matched_edit_idx"])
                for k in self.locator_k_list:
                    if prediction["rank"] <= k:
                        match_counts[k] += 1
                matched_edit_idxs.append(prediction["matched_edit_idx"])
                service_for_matched_edit[prediction["matched_edit_idx"]] = prediction["lsp_service"]
       
        # Update instance variables and log
        for k in self.locator_k_list:
            getattr(self, f"num_match_@{k}").append(match_counts[k])
            # print(f"==> Locator predicted MATCHED {match_counts[k]} edits in top {k}")
            
        return matched_edit_idxs, service_for_matched_edit, rank1_matched_edit_idxs
    
    def register_allowed_locator_match(self, predictions: list[dict], allowed_edit_idxs: list[int]|None):
        match_counts = {k: 0 for k in self.locator_k_list}
        for prediction in predictions:
            if prediction["matched"] and (allowed_edit_idxs is None or prediction["matched_edit_idx"] in allowed_edit_idxs):
                for k in self.locator_k_list:
                    if prediction["rank"] <= k:
                        match_counts[k] += 1
        
        for k in self.locator_k_list:
            getattr(self, f"num_allowed_match_@{k}").append(match_counts[k])
            # print(f"==> Locator predicted MATCHED {match_counts[k]} edits in top {k}")
    
    def register_edit_solution(self, edit_solutions, gold_solution):
        bleu_scores = []
        for solution in edit_solutions:
            (goldMap,predictionMap) = bleu.direct_computeMaps(solution, "".join(gold_solution))
            bleu_score = bleu.bleuFromMaps(goldMap, predictionMap)[0]
            bleu_scores.append(bleu_score)
            
        for k in self.generator_k_list:
            getattr(self, f"bleu_@{k}").append(max(bleu_scores[:k]))
            
        return getattr(self, f"bleu_@{self.generator_k_list[-1]}")[-1]
    
    def to_dict(self) -> dict:
        """
        Convert this SimulationRecord object to a dictionary for JSON serialization.
        """
        record_dict = {
            "commit_url": self.commit_url,
            "simulation_order": self.simulation_order,
            "locator_runtime": self.locator_runtime,
            "generator_runtime": self.generator_runtime,
        }
        
        # Add locator match statistics
        for k in self.locator_k_list:
            record_dict[f"num_match_@{k}"] = getattr(self, f"num_match_@{k}")
            record_dict[f"num_allowed_match_@{k}"] = getattr(self, f"num_allowed_match_@{k}")
        
        # Add generator BLEU scores
        for k in self.generator_k_list:
            record_dict[f"bleu_@{k}"] = getattr(self, f"bleu_@{k}")
            
        return record_dict
    
    def check(self) -> bool:
        assert len(self.simulation_order) - 1 == len(self.locator_runtime)
        assert len(self.locator_runtime) == len(self.generator_runtime)
        for k in self.locator_k_list:
            assert len(self.generator_runtime) == len(getattr(self, f"num_match_@{k}"))
            assert len(self.generator_runtime) == len(getattr(self, f"num_allowed_match_@{k}"))
            
        for k in self.generator_k_list:
            assert len(self.generator_runtime) == len(getattr(self, f"bleu_@{k}"))
        
def statistics_analyzer(records: list[SimulationRecord], args: argparse.Namespace, logger: logging.Logger):
    # Saving the statistics to a jsonl file
    os.makedirs(args.output_dir, exist_ok=True)
    json_records = []
    with jsonlines.open(os.path.join(args.output_dir, args.simulation_start_time + ".jsonl"), mode='w') as writer:
        for record in records:
            writer.write(record.to_dict())
            json_records.append(record.to_dict())
    
    if json_records == []:
        logger.info("No records found")
        return
    
    print_statistics(json_records, logger)
    
def print_statistics(json_records, logger):
    # For locator, it would rarely provide all K recommendations, the match count looks much lower
    # Hence we transform the match count to the percentage of having at least one match
    stats = {
        "num_records": len(json_records),
        "locator_runtime": [],
        "generator_runtime": [],
    }
    
    locator_k_list = SimulationRecord(None, None).locator_k_list
    generator_k_list = SimulationRecord(None, None).generator_k_list
    
    for k in locator_k_list:
        stats[f"hit_rate_@{k}"] = []
    for k in locator_k_list:
        stats[f"allowed_hit_rate_@{k}"] = []
        
    for k in generator_k_list:
        stats[f"bleu_@{k}"] = []
    
    for record in json_records:
        stats["locator_runtime"].extend(record["locator_runtime"])
        stats["generator_runtime"].extend(record["generator_runtime"])
        
        for k in locator_k_list:
            stats[f"hit_rate_@{k}"].extend([1 if i > 0 else 0 for i in record[f"num_match_@{k}"]])
            stats[f"allowed_hit_rate_@{k}"].extend([1 if i > 0 else 0 for i in record[f"num_allowed_match_@{k}"]])
        
        for k in generator_k_list:
            stats[f"bleu_@{k}"].extend(record[f"bleu_@{k}"])
            
    for key, performance in stats.items():
        if key == f"bleu_@{generator_k_list[-1]}":
            count_100 = sum(1 for x in performance if x == 100) / len(performance)
            count_50_100 = sum(1 for x in performance if 50 <= x < 100) / len(performance)
            count_less_50 = sum(1 for x in performance if x < 50) / len(performance)
        if isinstance(performance, list):
            stats[key] = sum(performance) / len(performance)
            
    stats["generator_accept_rate"] = count_100
    stats["generator_modify_rate"] = count_50_100
    stats["generator_reject_rate"] = count_less_50
    
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"\t{key}: {value:.4f}")
        else:
            logger.info(f"\t{key}: {value}")
        
    
if __name__ == "__main__":
    file_path = ""
    with jsonlines.open(file_path, mode='r') as reader:
        json_records = [record for record in reader]
    
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%Y/%m/%d %H:%M:%S',
                    level = logging.INFO)
    logger = logging.getLogger(__name__)
    print_statistics(json_records, logger)
    