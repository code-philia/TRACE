import os
import json
import time
import random
from commit import Commit
from record import SimulationRecord
from graph2seq import dfs_with_constraints
from Locators import predict_next_location, detect_next_clone
from Generators import generate_edit_solution

def simulation(test_sample, args, lang, logger, models):
    # STEP 1: get snapshot & edits of given commit url
    commit = Commit(test_sample["commit_url"])
    commit.language = lang
    
    # STEP 2: pick the init edit
    if test_sample["edit_order_graph"] is not None and not args.random_order:
        assert len(test_sample["edit_order_graph"]) == commit.hunk_num()
        allowed_simulation_order = dfs_with_constraints(test_sample["edit_order_graph"])
        allowed_init_edit_idxs = [order[0] for order in allowed_simulation_order]
    elif test_sample["edit_order_seq"] != [] and not args.random_order:
        for order in test_sample["edit_order_seq"]:
            assert len(order) == commit.hunk_num()
        allowed_simulation_order = test_sample["edit_order_seq"]
        allowed_init_edit_idxs = [order[0] for order in allowed_simulation_order]
    else:
        # Purely random order
        allowed_init_edit_idxs = list(range(commit.hunk_num()))
    
    """
    Edit dictionary format:
    {
        "idx": int,
        "before": list[str],
        "after": list[str],
        "simulated": bool
    }
    """
    init_edit_idx = 0 # random.choice(allowed_init_edit_idxs)
    if args.random_order:
        random_simulation_order = allowed_init_edit_idxs.copy()
        random_simulation_order.remove(init_edit_idx)
        random_simulation_order.sort()
        
    logger.info(f"Init edit idx: {init_edit_idx}")
    init_edit = commit.get_edit(init_edit_idx)
    init_edit["simulated"] = True
    commit.add_prev_edit(init_edit)
    record = SimulationRecord(test_sample["commit_url"], args)
    record.simulation_order.append(init_edit_idx)
    
    # STEP 3: start simulation
    if args.system in ["TRACE", "TRACE-wo-Invoker"]:
        if commit.language == "python":
            from LSPs.py_lsp import PyLanguageServer
            LSP = PyLanguageServer(args.lsp_log)
        elif commit.language == "java":
            from LSPs.java_lsp import JavaLanguageServer
            LSP = JavaLanguageServer(args.lsp_log)
        elif commit.language == "go":
            from LSPs.go_lsp import GoLanguageServer
            LSP = GoLanguageServer(args.lsp_log)
        elif commit.language in ["javascript", "typescript"]:
            from LSPs.jsts_lsp import TsLanguageServer
            LSP = TsLanguageServer(commit.language, args.lsp_log)
        
        files_to_change = [os.path.join(commit.project_dir, file_path) for file_path in commit.changed_files]

        # Initialize LSP
        max_retries = 5
        retry_delay = 10 
        for attempt in range(1, max_retries + 1):
            try:
                LSP.initialize(commit.project_dir)
                break  # If no exception, break the loop
            except Exception as e:
                print(f"[Attempt {attempt}] Error initializing LSP: {e}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Failed to initialize LSP after multiple attempts.")
                    raise  # If all attempts failed, raise the exception

        LSP.open_in_batch(files_to_change)
        # Obtain the initial diagnose messages that can be ignored
        init_diagnose = LSP.process_diagnose(commit, models, args, return_diagnose=True)
        args.init_diagnose_msg = LSP.extract_diagnose_msg(init_diagnose)
    else:
        LSP = None
    
    while len(commit.prev_edits) != commit.hunk_num(): # simulate each edit
        # STEP 3.1: predict the edit operation label for each line of code
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.info("> Finding the next edit location")
        logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        record.locator_runtime.append(0) # Start predicting the next edit location
        if args.system == "CodeCloneDetector":
            location_predictions = detect_next_clone(commit, record)
        else:
            location_predictions = predict_next_location(commit, models, args, record, logger, LSP=LSP)
        
        # STEP 3.2: acquire the gold labels for current snapshots
        gold_labels = commit.get_gold_labels(args)

        # STEP 3.3: compare with the ground truth
        # STEP 3.3.1: Check precision, recall and F1 score for predicted labels
        # TODO: Implement this
        
        # STEP 3.3.2: Check if the predicted locations match with the ground truth hunk
        consecutive_location_predictions = check_locator_match(location_predictions, gold_labels, commit, args)
        # Only keep the consecutive actionable locations, ignore the rest non-actionable locations (e.g. <null>, <keep>)
        # May contain multiple rank 1 edit location recommendations
        matched_edit_idxs, service_for_matched_edit, rank1_matched_edit_idxs = record.register_locator_match(consecutive_location_predictions, args, logger)
        
        # STEP 3.4: pick the next edit
        if (test_sample["edit_order_graph"] is not None or test_sample["edit_order_seq"] != []) and not args.random_order:
            # based on previous simulation order, shrink down allowed_simulation_order
            last_edit_idx = commit.prev_edits[-1]["idx"]
            # pop the first element of allowed_simulation_order, as it is the last simulated edit
            allowed_simulation_order = [order[1:] for order in allowed_simulation_order if order[0] == last_edit_idx]
            allowed_next_edit_idxs = [order[0] for order in allowed_simulation_order]
            # retain the order of allowed_next_edit_idxs, but only keep the idxs that are in matched_edit_idxs and allowed_next_edit_idxs
            logger.info(f"==> Allowed next edit idxs: {allowed_next_edit_idxs}")
            logger.info(f"==> Matched edit idxs: {matched_edit_idxs}")
            prediced_and_allowed_next_edit_idxs = [idx for idx in allowed_next_edit_idxs if idx in set(matched_edit_idxs).intersection(set(allowed_next_edit_idxs))]
            logger.info(f"==> Prediced & allowed next edit idxs: {prediced_and_allowed_next_edit_idxs}")
            if prediced_and_allowed_next_edit_idxs:
                logger.info(f"==> Locator prediction MATCHED {len(prediced_and_allowed_next_edit_idxs)} edits: {prediced_and_allowed_next_edit_idxs}")
                simulating_edit_idx = prediced_and_allowed_next_edit_idxs[0]
                simulating_edit_lsp_service = service_for_matched_edit[simulating_edit_idx]
                logger.info(f"==> Location found via LSP service: {simulating_edit_lsp_service}")
            else:
                logger.info(f"==> Locator prediction MISSED next edits, due to order constraints or no matched edits")
                simulating_edit_idx = random.choice(allowed_next_edit_idxs)
                simulating_edit_lsp_service = "normal"
            record.register_allowed_locator_match(consecutive_location_predictions, prediced_and_allowed_next_edit_idxs)
        else:
            if matched_edit_idxs:
                logger.info(f"==> Locator predicted MATCHED {len(matched_edit_idxs)} edits: {matched_edit_idxs}")
                simulating_edit_lsp_service = service_for_matched_edit[matched_edit_idxs[0]]
                logger.info(f"==> Location found via LSP service: {simulating_edit_lsp_service}")
                if len(rank1_matched_edit_idxs) <= 1:
                    next_simulating_edit_idxs = [matched_edit_idxs[0]]
                else:
                    next_simulating_edit_idxs = rank1_matched_edit_idxs
            else:
                logger.info(f"==> Locator prediction MISSED next edits, due to no matched edits")
                while True: # Continue to pop until we find an edit that is not simulated
                    next_simulating_edit_idxs = [random_simulation_order.pop(0)]
                    if next_simulating_edit_idxs[0] in record.simulation_order:
                        continue
                    else:
                        break
                simulating_edit_lsp_service = "normal"
            record.register_allowed_locator_match(consecutive_location_predictions, None)
        
        for idx, simulating_edit_idx in enumerate(next_simulating_edit_idxs):
            if idx != 0:
                # this extra predcited location will skip locator
                record.locator_runtime.append(0)
                for k in record.locator_k_list:
                    getattr(record, f"num_match_@{k}").append(1)
                    if args.random_order:
                        getattr(record, f"num_allowed_match_@{k}").append(1)
            logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"> Generating edit solution on edit {simulating_edit_idx}")
            logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            if simulating_edit_idx in record.simulation_order:
                raise ValueError(f"Repetitive simulation of edit hunk {simulating_edit_idx} at commit {commit.commit_url}")
            record.simulation_order.append(simulating_edit_idx)
            # STEP 3.5: generate edit solution based on picked next edit location
            record.generator_runtime.append(0)
            edit_solutions = generate_edit_solution(commit, simulating_edit_idx, simulating_edit_lsp_service, location_predictions, args, models, record, logger)
            
            # STEP 3.6: compare with the ground truth
            simulating_edit = commit.get_edit(simulating_edit_idx)
            best_bleu_score = record.register_edit_solution(edit_solutions, simulating_edit["after"])
            logger.info(f"==> Generator predicted BLEU@10: {best_bleu_score}\n")
            
            # STEP 3.7: add this edit to simulated edits
            simulating_edit = commit.get_edit(simulating_edit_idx)
            simulating_edit["simulated"] = True
            commit.add_prev_edit(simulating_edit)
    
    try:
        if LSP:
            LSP.close()
    except:
        pass
    
    return record
        
def check_locator_match(location_predictions, gold_labels, commit, args):
    """
    Check how many predicted locations match with the ground truth
    """
    # STEP 1: combine the consecutive predicted locations into a single location
    predicted_consecutive_locations = combine_consecutive_locations(location_predictions, data_type="predicted", args=args)
    gold_consecutive_locations = combine_consecutive_locations(gold_labels, data_type="gold", args=args)
    unsimulated_locations = commit.unsimulated_edit_locations(args)
    try:
        assert len(unsimulated_locations) == len(gold_consecutive_locations)
    except:
        error_info = {
            "unsimulated_locations": unsimulated_locations,
            "gold_consecutive_locations": gold_consecutive_locations
        }
        with open("error.json", "w") as f:
            json.dump(error_info, f, indent=4)
        with open("reproduce_error_env.json", "w") as f:
            json.dump(gold_labels, f, indent=4)
        raise ValueError(f"Have {len(unsimulated_locations)} unsimulated location, but have {len(gold_consecutive_locations)} gold consecutive locations")
        
    # assign the edit hunk idx to gold_consecutive_locations
    for gold_location in gold_consecutive_locations:
        for unsimulated_location in unsimulated_locations:
            if gold_location["file_path"] == unsimulated_location["file_path"] and gold_location["line_idxs"] == unsimulated_location["line_idxs"]:
                gold_location["hunk_idx"] = unsimulated_location["hunk_idx"]
                break
        try:
            assert "hunk_idx" in gold_location
        except:
            with open("gold_consecutive_locations.json", "w") as f:
                json.dump(gold_consecutive_locations, f, indent=4)
            with open("unsimulated_locations.json", "w") as f:
                json.dump(unsimulated_locations, f, indent=4)
            raise Exception("Stop here")
    
    # STEP 2: check each predicted location, whether it matches with the ground truth
    # Pay attention to the case where multiple predicted locations match with the same ground truth location
    for predicted_location in predicted_consecutive_locations:
        for gold_location in gold_consecutive_locations:
            if "matched" in gold_location and gold_location["matched"]:
                # If this gold location has already been matched, skip, avoid double counting
                continue
            if predicted_location["file_path"] != gold_location["file_path"]:
                # If the predicted location is not in the same file as the gold location, skip
                continue
            if gold_location["type"] == "inter" and predicted_location["type"] == "inter":
                if predicted_location["line_idxs"][0] - 1 <= gold_location["line_idxs"][0] <= predicted_location["line_idxs"][0] + 1: # Give 1 line of tolerance
                    predicted_location["matched"] = True
                    predicted_location["matched_edit_idx"] = gold_location["hunk_idx"]
                    predicted_location["matched_overlap_ratio"] = 1
                    gold_location["matched"] = True
            else:
                overlap_set = set(predicted_location["line_idxs"]).intersection(set(gold_location["line_idxs"]))
                overlap_ratio = 2 * len(overlap_set) / (len(predicted_location["line_idxs"]) + len(gold_location["line_idxs"]))
                if overlap_ratio > 0.5:
                    predicted_location["matched"] = True
                    predicted_location["matched_edit_idx"] = gold_location["hunk_idx"]
                    predicted_location["matched_overlap_ratio"] = overlap_ratio
                    gold_location["matched"] = True
        
        predicted_location["abs_file_path"] = os.path.join(commit.project_dir, predicted_location["file_path"])
        if "matched" not in predicted_location:
            predicted_location["matched"] = False
            predicted_location["matched_edit_idx"] = None
            predicted_location["matched_overlap_ratio"] = 0
    
    # STEP 3: return
    return predicted_consecutive_locations
    
def combine_consecutive_locations(location_predictions, data_type, args):
    """
    Combine the consecutive predicted locations into a single location
    
    Args:
        location_predictions: dict, each key is a file, each value is another dict, keys including: "inline_predictions","inline_confidences", "inter_predictions", "inter_confidences"
        data_type: str, "predicted" or "gold". If gold, skip all confidence calculation
        args: argparse.Namespace, the arguments of the simulation
    
    Returns:
        combined_locations: list, each element is a dict
    """
    lsp_service_rank = { # the recommendation order
        'rename': 4,
        'def&use': 3,
        'clone': 2,
        'diagnose': 1,
        'normal': 0
    }
    if args.label_num == 6:
        # For TRACE, TRACE-wo-Invoker, EnrichedSemantics
        combined_locations = []
        
        for file_path, file_predictions in location_predictions.items():
            if data_type == "predicted":
                inter_predictions = file_predictions["inter_predictions"]
                inter_confidences = file_predictions["inter_confidences"]
                inline_predictions = file_predictions["inline_predictions"]
                inline_confidences = file_predictions["inline_confidences"]
                inline_service = file_predictions["inline_service"]
                inter_service = file_predictions["inter_service"]
            else:
                inter_predictions = file_predictions["inter_golds"]
                inline_predictions = file_predictions["inline_golds"]
            
            # Assert the element in inter_predictions and inline_predictions are allowed
            assert set(inter_predictions).issubset({"<null>", "<block-split>", "<insert>"})
            assert set(inline_predictions).issubset({"<keep>", "<delete>", "<replace>"})
            
            # Zip inter-line predictions with inline predictions into a zipped single list of prediction
            zipped_predictions = [inter_predictions[0]]
            for inter, inline in zip(inter_predictions[1:], inline_predictions):
                zipped_predictions.append(inline)
                zipped_predictions.append(inter)
            zipped_pred_actionable_idxs = [i for i, action in enumerate(zipped_predictions) if action != "<keep>" and action != "<null>"]
            
            if data_type == "predicted":
                zipped_confidences = [inter_confidences[0]]
                for inter, inline in zip(inter_confidences[1:], inline_confidences):
                    zipped_confidences.append(inline)
                    zipped_confidences.append(inter)

                # Set confidence of un-actionable locations to 0, as we only rank actionable locations
                zipped_confidences = [confidence if i in zipped_pred_actionable_idxs else 0 for i, confidence in enumerate(zipped_confidences)]
                
                zipped_services = [inter_service[0]]
                for inter, inline in zip(inter_service[1:], inline_service):
                    zipped_services.append(inline)
                    zipped_services.append(inter)
            
            # TYPE 1: Extract inline consecutive edit locations
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            inline_edit_location_groups = []
            location_group = []
            for line_idx, label in enumerate(inline_predictions):
                if label == "<keep>" and location_group != []:
                    inline_edit_location_groups.append(location_group)
                    location_group = []
                elif label != "<keep>":
                    location_group.append(line_idx)
            if location_group != []:
                inline_edit_location_groups.append(location_group)
            
            # Add more information to the grouped locations
            for group in inline_edit_location_groups:
                start_line_idx = group[0]
                end_line_idx = group[-1]
                
                zipped_start_idx = start_line_idx * 2
                zipped_end_idx = end_line_idx * 2 + 2
                
                group_labels = zipped_predictions[zipped_start_idx:zipped_end_idx+1]
                if data_type == "predicted":
                    # Calculate the average confidence of the group (ignore 0 confidence, those are un-actionable locations)
                    group_confidences = zipped_confidences[zipped_start_idx:zipped_end_idx+1]
                    avg_confidence = sum(group_confidences) / len([x for x in group_confidences if x != 0])

                    # Find out which lsp service is used to locate this group
                    group_services = zipped_services[zipped_start_idx:zipped_end_idx+1]
                    lsp_service = list(set(group_services).difference({"normal"}))
                    if len(lsp_service) == 0:
                        lsp_service = "normal"
                    elif len(lsp_service) == 1:
                        lsp_service = lsp_service[0]
                    else:
                        lsp_service = " + ".join(lsp_service)
                        # raise Exception(f"More than 1 lsp service is used to locate this group: {lsp_service}")
                    
                    combined_locations.append({
                        "file_path": file_path,
                        "line_idxs": group,
                        "zipped_labels": group_labels,
                        "confidence": 1 if lsp_service != "normal" else avg_confidence,
                        "type": "both",
                        "lsp_service": lsp_service
                    })
                else:
                    combined_locations.append({
                        "file_path": file_path,
                        "line_idxs": group,
                        "zipped_labels": group_labels,
                        "type": "both"
                    })
            grouped_idx = [item for sublist in inline_edit_location_groups for item in sublist] # flatten the list, a list with lines labelled to edit
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            # TYPE 2: Extract inter-line edit locations (inter-line edits either a part of inline consecutive group or a standalone inter-line edit)
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            for inter_idx, inter_label in enumerate(inter_predictions):
                if inter_label == "<insert>":
                    before_inline_idx = None if inter_idx == 0 else inter_idx - 1
                    after_inline_idx = None if inter_idx == len(inter_predictions) - 1 else inter_idx
                    if (before_inline_idx is not None and before_inline_idx in grouped_idx) or \
                    (after_inline_idx is not None and after_inline_idx in grouped_idx):
                        # in this case we dont need to add this insert into the grouped locations, as it is already included in the inline consecutive group
                        continue
                    else:
                        # This is a standalone inter-line edit
                        if data_type == "predicted":
                            combined_locations.append({
                                "file_path": file_path,
                                "line_idxs": [inter_idx],
                                "zipped_labels": [inter_label],
                                "confidence": inter_confidences[inter_idx],
                                "type": "inter",
                                "lsp_service": inter_service[inter_idx]
                            })
                        else:
                            combined_locations.append({
                                "file_path": file_path,
                                "line_idxs": [inter_idx],
                                "zipped_labels": [inter_label],
                                "type": "inter"
                            })
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        if data_type == "predicted":
            # rank predicted consectuive locations by their confidence
            combined_locations = sorted(
                combined_locations,
                key=lambda x:(
                    x["confidence"],
                    lsp_service_rank.get(x["lsp_service"], 0)
                ),
                reverse=True
            )
            prev_conf = 1
            rank = 1
            for idx, location in enumerate(combined_locations):
                if location["confidence"] == prev_conf:
                    location["rank"] = rank
                elif location["confidence"] < prev_conf:
                    if idx != 0:
                        rank += 1
                    location["rank"] = rank
                    prev_conf = location["confidence"]
                else:
                    raise ValueError("Confidence values are not in expected order.")
            
        return combined_locations
    
    elif args.label_num == 3 and args.system != "CoEdPilot":
        # For PlainSemantics and CodeCloneDetector
        combined_locations = []
        
        for file_path, file_predictions in location_predictions.items():
            if data_type == "predicted":
                inline_predictions = file_predictions["inline_predictions"]
                inline_confidences = file_predictions["inline_confidences"]
            else:
                inline_predictions = file_predictions["inline_golds"]
                
            assert set(inline_predictions).issubset({"<keep>", "<insert>", "<replace>"})
            
            # TYPE 1: Extract inline consecutive edit locations
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            inline_edit_location_groups = []
            location_group = []
            last_label = "<keep>"
            for line_idx, label in enumerate(inline_predictions):
                if last_label == "<keep>" and label != "<keep>":
                    # e.g. <keep> -> <replace> or <keep> -> <add>
                    location_group.append(line_idx)
                    last_label = label
                elif last_label != "<keep>" and label == "<keep>":
                    # e.g. <replace> -> <keep> or <add> -> <keep>
                    inline_edit_location_groups.append(location_group)
                    location_group = []
                    last_label = label
                elif last_label != "<keep>" and label != "<keep>":
                    if last_label == label and label == "<replace>":
                        # e.g. <replace> -> <replace> 
                        location_group.append(line_idx)
                    else: 
                        # e.g. <replace> -> <add>
                        # e.g. <add> -> <add>, this means that 2 consecutive lines of code all require adding new lines of code after each of them
                        inline_edit_location_groups.append(location_group)
                        location_group = []
                        location_group.append(line_idx)
                        last_label = label
                else:
                    continue
                    
            if location_group != []:
                inline_edit_location_groups.append(location_group)
                
            # Add more information to the grouped locations
            for group in inline_edit_location_groups:
                start_line_idx = group[0]
                end_line_idx = group[-1]
                
                group_labels = inline_predictions[start_line_idx:end_line_idx+1]
                if data_type == "predicted":
                    group_confidences = inline_confidences[start_line_idx:end_line_idx+1]
                    avg_confidence = sum(group_confidences) / len([x for x in group_confidences if x != 0])
                    combined_locations.append({
                        "file_path": file_path,
                        "line_idxs": group,
                        "zipped_labels": group_labels,
                        "confidence": avg_confidence,
                        "type": "both",
                        "lsp_service": "normal"
                    })
                else:
                    combined_locations.append({
                        "file_path": file_path,
                        "line_idxs": group,
                        "zipped_labels": group_labels,
                        "type": "both"
                    })
        
        if data_type == "predicted":
            # rank predicted consectuive locations by their confidence
            combined_locations = sorted(
                combined_locations,
                key=lambda x:(
                    x["confidence"],
                    lsp_service_rank.get(x["lsp_service"], 0)
                ),
                reverse=True
            )
            prev_conf = 1
            rank = 1
            for idx, location in enumerate(combined_locations):
                if location["confidence"] == prev_conf:
                    location["rank"] = rank
                elif location["confidence"] < prev_conf:
                    if idx != 0:
                        rank += 1
                    location["rank"] = rank
                    prev_conf = location["confidence"]
                else:
                    raise ValueError("Confidence values are not in expected order.")
            
        return combined_locations
    
    elif args.label_num == 3 and args.system == "CoEdPilot":
        # For CoEdPilot only
        combined_locations = []
        
        for file_path, file_predictions in location_predictions.items():
            if data_type == "predicted":
                inline_predictions = file_predictions["inline_predictions"]
                inline_confidences = file_predictions["inline_confidences"]
            else:
                inline_predictions = file_predictions["inline_golds"]
            
            assert set(inline_predictions).issubset({"<keep>", "<add>", "<replace>"})
        
            # TYPE 1: Extract inline consecutive edit locations
            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            inline_edit_location_groups = []
            location_group = []
            last_label = "<keep>"
            for line_idx, label in enumerate(inline_predictions):
                if last_label == "<keep>" and label != "<keep>":
                    # e.g. <keep> -> <replace> or <keep> -> <add>
                    location_group.append(line_idx)
                    last_label = label
                elif last_label != "<keep>" and label == "<keep>":
                    # e.g. <replace> -> <keep> or <add> -> <keep>
                    inline_edit_location_groups.append(location_group)
                    location_group = []
                    last_label = label
                elif last_label != "<keep>" and label != "<keep>":
                    if last_label == label and label == "<replace>":
                        # e.g. <replace> -> <replace> (<add> -> <add> is unlikely, as <add> is suppose to be single line edit)
                        location_group.append(line_idx)
                    else: 
                        # e.g. <replace> -> <add>
                        # e.g. <add> -> <add>, this means that 2 consecutive lines of code all require adding new lines of code after each of them
                        inline_edit_location_groups.append(location_group)
                        location_group = []
                        location_group.append(line_idx)
                        last_label = label
                else:
                    continue
                
            if location_group != []:
                inline_edit_location_groups.append(location_group)
            
            # Add more information to the grouped locations
            for group in inline_edit_location_groups:
                start_line_idx = group[0]
                end_line_idx = group[-1]
                
                group_labels = inline_predictions[start_line_idx:end_line_idx+1]
                if data_type == "predicted":
                    group_confidences = inline_confidences[start_line_idx:end_line_idx+1]
                    avg_confidence = sum(group_confidences) / len([x for x in group_confidences if x != 0])
                    combined_locations.append({
                        "file_path": file_path,
                        "line_idxs": group,
                        "zipped_labels": group_labels,
                        "confidence": avg_confidence,
                        "type": "both",
                        "lsp_service": "normal"
                    })
                else:
                    combined_locations.append({
                        "file_path": file_path,
                        "line_idxs": group,
                        "zipped_labels": group_labels,
                        "type": "both"
                    })
                
        if data_type == "predicted":
            # rank predicted consectuive locations by their confidence
            combined_locations = sorted(
                combined_locations,
                key=lambda x:(
                    x["confidence"],
                    lsp_service_rank.get(x["lsp_service"], 0)
                ),
                reverse=True
            )
            prev_conf = 1
            rank = 1
            for idx, location in enumerate(combined_locations):
                if location["confidence"] == prev_conf:
                    location["rank"] = rank
                elif location["confidence"] < prev_conf:
                    if idx != 0:
                        rank += 1
                    location["rank"] = rank
                    prev_conf = location["confidence"]
                else:
                    raise ValueError("Confidence values are not in expected order.")
            
        return combined_locations
