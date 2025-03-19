import time
import json
import torch

from commit import Commit
from rank_bm25 import BM25Okapi
from code_window import CodeWindow
from torch.utils.data import TensorDataset, DataLoader
from enriched_semantic import construct_prior_edit_hunk
from utils import label_conversion, label_conversion_reverse
from CoEdPilot_estimator import load_estimator_data, evaluate_estimator
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer

def load_generator(args, logger):
    config_class, model_class, tokenizer_class = (T5Config, T5ForConditionalGeneration, RobertaTokenizer)
    
    config = config_class.from_pretrained("salesforce/codet5-base")
    tokenizer = tokenizer_class.from_pretrained("salesforce/codet5-base")
    model = model_class.from_pretrained("salesforce/codet5-base")
    new_special_tokens = ["<inter-mask>",
                          "<code_window>", "</code_window>", 
                          "<prompt>", "</prompt>", 
                          "<prior_edits>", "</prior_edits>",
                          "<edit>", "</edit>",
                          "<keep>", "<replace>", "<delete>",
                          "<null>", "<insert>", "<block-split>",
                          "<replace-by>", "</replace-by>",
                          "<feedback>", "</feedback>"]
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    model.encoder.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    model.load_state_dict(torch.load(args.generator_model_path))
    model.to(args.device)
    logger.info(f"Successfully loaded generator model from: {args.generator_model_path}")
    return model, tokenizer

def generate_edit_solution(commit, simulating_edit_idx, lsp_service, location_predictions, args, models, record, logger):
    """
    Given a edit location, commit message (natural language description) and prior edits, generate the edit solution
    """
    generator = models["generator"]
    generator_tokenizer = models["generator_tokenizer"]
    target_edit = commit.get_edit(simulating_edit_idx)
    edit_at_file = commit.map[target_edit["idx"]]["at_file"]
    snapshot = commit.snapshots[edit_at_file]
    target_edit_hunk = construct_prior_edit_hunk(snapshot, target_edit, commit.language)
        
    if args.system != "CoEdPilot":
        if not args.label_correction and args.label_num == 6:
            # if user does not correct the label, the input label for generator is the predicted label
            unsimulated_locations = commit.unsimulated_edit_locations(args)
            
            for unsimulated_location in unsimulated_locations:
                if unsimulated_location["hunk_idx"] == target_edit["idx"]:
                    target_edit_location = unsimulated_location["line_idxs"]

            tgt_hunk_prefix_len = target_edit_hunk["prefix_len"]
            tgt_hunk_suffix_len = target_edit_hunk["suffix_len"]
            if target_edit_hunk["before_edit"] != []: # replace / delete type
                offset = 1
            elif target_edit_hunk["before_edit"] == []: # insert type
                offset = 0
            inline_predicted_labels = location_predictions[edit_at_file]["inline_predictions"][target_edit_location[0]-tgt_hunk_prefix_len: target_edit_location[-1] + tgt_hunk_suffix_len + offset]
            inter_predicted_labels = location_predictions[edit_at_file]["inter_predictions"][target_edit_location[0]-tgt_hunk_prefix_len: target_edit_location[-1] + tgt_hunk_suffix_len + 1 + offset]
            
            # assert the length is correct
            assert len(inline_predicted_labels) == len(target_edit_hunk["inline_labels"])
            assert len(inline_predicted_labels) == len(inter_predicted_labels) - 1
            
            inline_predicted_labels = [label[1:-1] if label[0] == "<" and label[-1] == ">" else label for label in inline_predicted_labels]
            inter_predicted_labels = [label[1:-1] if label[0] == "<" and label[-1] == ">" else label for label in inter_predicted_labels]
            if inline_predicted_labels == target_edit_hunk["inline_labels"] and inter_predicted_labels == target_edit_hunk["inter_labels"]:
                logger.info(f"Label prediction is PERFECT match")
            else:
                logger.info(f"Label prediction is NOT PERFECT match")
            
            # overwrite the predicted labels with the gold labels
            target_edit_hunk["inline_labels"] = inline_predicted_labels
            target_edit_hunk["inter_labels"] = inter_predicted_labels
        
        elif not args.label_correction and args.label_num == 3:
            # For PlainSemantics and CodeCloneDetector
            unsimulated_locations = commit.unsimulated_edit_locations(args)
            for unsimulated_location in unsimulated_locations:
                if unsimulated_location["hunk_idx"] == target_edit["idx"]:
                    target_edit_location = unsimulated_location["line_idxs"]
                    
            tgt_hunk_prefix_len = target_edit_hunk["prefix_len"]
            tgt_hunk_suffix_len = target_edit_hunk["suffix_len"]

            predicted_labels = location_predictions[edit_at_file]["inline_predictions"][target_edit_location[0]-tgt_hunk_prefix_len: target_edit_location[-1] + tgt_hunk_suffix_len + 1]
            
            # assert the length is correct
            assert len(predicted_labels) == len(target_edit_hunk["inline_labels"])
            
            # convert 6 label to 3 label
            gold_labels = label_conversion(target_edit_hunk["inline_labels"], target_edit_hunk["inter_labels"])
            
            predicted_labels = [label[1:-1] if label[0] == "<" and label[-1] == ">" else label for label in predicted_labels]
            
            if predicted_labels == gold_labels:
                logger.info(f"Label prediction is PERFECT match")
            else:
                logger.info(f"Label prediction is NOT PERFECT match")
                
            inline_predicted_labels, inter_predicted_labels = label_conversion_reverse(predicted_labels)
            target_edit_hunk["inline_labels"] = inline_predicted_labels
            target_edit_hunk["inter_labels"] = inter_predicted_labels
        
        target_edit_hunk = CodeWindow(target_edit_hunk, "hunk")
        selected_prior_edits = TRACE_select_prior_edits(target_edit_hunk, commit.enriched_prev_edits, generator_tokenizer)
        input_dataset = TRACE_formalize_generator_input(target_edit_hunk, commit.commit_message, lsp_service, selected_prior_edits, models["generator_tokenizer"], args)
    
    elif args.system == "CoEdPilot":
        # For CoEdPilot
        # convert gold (6 labels) to CoEdPilot labels
        gold_labels = label_conversion(target_edit_hunk["inline_labels"], target_edit_hunk["inter_labels"])
        for i, label in enumerate(gold_labels):
            gold_labels[i] = label.replace("insert", "add")
            
        if not args.label_correction:
            unsimulated_locations = commit.unsimulated_edit_locations(args)
            for unsimulated_location in unsimulated_locations:
                if unsimulated_location["hunk_idx"] == target_edit["idx"]:
                    target_edit_location = unsimulated_location["line_idxs"]
            
            
            tgt_hunk_prefix_len = target_edit_hunk["prefix_len"]
            tgt_hunk_suffix_len = target_edit_hunk["suffix_len"]

            if target_edit_hunk["before_edit"] != []: # replace / delete type
                offset = 1
            elif target_edit_hunk["before_edit"] == []: # insert type
                offset = 0
            predicted_labels = location_predictions[edit_at_file]["inline_predictions"][target_edit_location[0]-tgt_hunk_prefix_len: target_edit_location[-1] + tgt_hunk_suffix_len + offset]
            
            # assert the length is correct
            assert len(predicted_labels) == len(target_edit_hunk["inline_labels"])

            if predicted_labels == gold_labels:
                logger.info(f"Label prediction is PERFECT match")
            else:
                logger.info(f"Label prediction is NOT PERFECT match")
            
            gold_labels = predicted_labels
        
        selected_prior_edits = CoEdPilot_select_prior_edits(target_edit_hunk, commit, models)
        input_dataset = CoEdPilot_formalize_generator_input(target_edit_hunk, gold_labels, commit.commit_message, selected_prior_edits, models["generator_tokenizer"])
        
    dataloader = DataLoader(input_dataset, batch_size=1, shuffle=False)
    start_time = time.time()
    generator.eval()
    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)

        source_ids = batch[0]
        source_mask = source_ids.ne(generator_tokenizer.pad_token_id)
        with torch.no_grad():
            preds = generator.generate(source_ids,
                                    attention_mask=source_mask,
                                    use_cache=True,
                                    num_beams=10,
                                    max_length=512,
                                    num_return_sequences=10)
            preds = preds.reshape(source_ids.size(0), 10, -1)
            preds = preds.cpu().numpy()
            for idx in range(preds.shape[0]):
                edit_solutions = []
                for candidate in preds[idx]:
                    edit_solutions.append(generator_tokenizer.decode(candidate, skip_special_tokens=True,clean_up_tokenization_spaces=False))
    end_time = time.time()
    record.generator_runtime[-1] += end_time - start_time
        
    
    return edit_solutions

def CoEdPilot_select_prior_edits(tgt_hunk: dict, commit: Commit, models: dict) -> list[dict]:
    """
    Func: 
        Given a target hunk and a list of other hunks, select the prior edits from the other hunks
    Args:
        tgt_hunk: dict, the enriched target hunk (not the CodeWindow)
        commit: Commit, the commit
        models: dict, the models
    Return:
        prior_edits: list[dict], the prior edits
    """
    estimator_dataset = []
    choosen_hunk_ids = []
    tgt_at_file = commit.map[tgt_hunk["idx"]]["at_file"]
    tgt_at_line = tgt_hunk["edit_start_line_idx"]
    for hunk in commit.enriched_prev_edits:
        choosen_hunk_ids.append(hunk.idx)
        if tgt_at_file == commit.map[hunk.idx]["at_file"]:
            code_distance = max(0, 1 - abs(tgt_at_line - hunk.edit_start_line_idx) / 50)
        else:
            code_distance = 0
        sample = {
            "sliding_window": tgt_hunk,
            "prior_edit": hunk,
            "code_distance": code_distance
        }
        estimator_dataset.append(sample)
    
    if len(estimator_dataset) == 0:
        return []
    
    estimator = models["estimator"]
    estimator_tokenizer = models["estimator_tokenizer"]
    dependency_tokenizer = models["dependency_tokenizer"]
    estimator_dataset = load_estimator_data(estimator_dataset, estimator_tokenizer, dependency_tokenizer)
    estimator_dataloader = DataLoader(estimator_dataset, batch_size=20, shuffle=False)
    # get the prior edit estimation
    estimation = evaluate_estimator(estimator, estimator_dataloader, "infer", show_progress=False)
    # project hunk id to estimation score
    scores = []
    for hunk_id, score in zip(choosen_hunk_ids, estimation):
        scores.append([hunk_id, score])
    # sort in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    top_3_prior_edits = scores[:3]
    prior_edits = []
    for hunk_id in top_3_prior_edits:
        prior_edits.extend([hunk for hunk in commit.prev_edits if hunk["idx"] == hunk_id[0]])
    
    return prior_edits
    
def TRACE_select_prior_edits(tgt_hunk: CodeWindow, prev_eidt_hunks: list[CodeWindow], tokenizer: RobertaTokenizer) -> list[CodeWindow]:
    """
    Func: 
        Given a target hunk and a list of other hunks, select the prior edits from the other hunks
    Args:
        tgt_hunk: CodeWindow, the target hunk
        prev_eidt_hunks: list[CodeWindow], the other hunks
        tokenizer: RobertaTokenizer, the tokenizer
    Return:
        prior_edits: list[CodeWindow], the prior edits
    """
    assert isinstance(tgt_hunk, CodeWindow)
    choosen_hunk_ids = [hunk.idx for hunk in prev_eidt_hunks] # index to hunk id
    tokenized_corpus = [tokenizer.tokenize("".join(hunk.before_edit_region()+hunk.after_edit_region())) for hunk in prev_eidt_hunks]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = tokenizer.tokenize(tgt_hunk.before_edit_window(split_by_line=False))
    retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3) 
    retrieved_index = [tokenized_corpus.index(i) for i in retrieval_code] # get index in choosen_hunk_ids
    prior_edit_id = [choosen_hunk_ids[idx] for idx in retrieved_index] # get corresponding hunk id
    prior_edits = []
    for id in prior_edit_id: # preserve the order
        prior_edits.append([hunk for hunk in prev_eidt_hunks if hunk.idx == id][0])
    
    return prior_edits

def CoEdPilot_formalize_generator_input(target_edit_hunk: dict, gold_labels: list[str], prompt: str, prior_edits: list[dict], tokenizer: RobertaTokenizer) -> str:
    source_seq = ""
    target_edit_hunk = CodeWindow(target_edit_hunk, "hunk")
    assert len(gold_labels) == len(target_edit_hunk.before_edit_window())
    for line_of_code, label in zip(target_edit_hunk.before_edit_window(), gold_labels):
        source_seq += f"{label} {line_of_code}"
        
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, max_length=64, truncation=True)
    truncated_prompt = tokenizer.decode(encoded_prompt)
    source_seq += f"</s>{truncated_prompt}</s> "
    for prior_edit in prior_edits:
        if prior_edit["before"] == []:
            prior_edit_seq = f"add {''.join(prior_edit['after'])} </s>"
        else:
            prior_edit_seq = f"remove {''.join(prior_edit['before'])} </s> add {''.join(prior_edit['after'])} </s>"
        source_seq += prior_edit_seq
    

    encoded_source_seq = tokenizer(source_seq, padding="max_length", truncation=True, max_length=512)
    source_ids = torch.tensor([encoded_source_seq["input_ids"]], dtype=torch.long)
    data = TensorDataset(source_ids)
    
    return data
        
def TRACE_formalize_generator_input(sliding_window: CodeWindow, prompt: str, lsp_service: str,
        prior_edits: list[CodeWindow], tokenizer, args) -> str:
    if args.system in ["TRACE", "TRACE-wo-Invoker"]:
        source_seq = f"<feedback>{lsp_service}</feedback>"
    else:
        source_seq = ""
    source_seq += sliding_window.formalize_as_generator_target_window(beautify=False, label_num=args.label_num)
    # prepare the prompt region
    # truncate prompt if it encode to more than 64 tokens
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, max_length=64, truncation=True)
    truncated_prompt = tokenizer.decode(encoded_prompt)
    source_seq += f"</code_window><prompt>{truncated_prompt}</prompt><prior_edits>"
    common_seq_len = len(tokenizer.encode(source_seq, add_special_tokens=False))
    # prepare the prior edits region
    for prior_edit in prior_edits:
        prior_edit_seq = prior_edit.formalize_as_prior_edit(beautify=False, label_num=args.label_num)
        prior_edit_seq_len = len(tokenizer.encode(prior_edit_seq, add_special_tokens=False))
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        source_seq += prior_edit_seq
        common_seq_len += prior_edit_seq_len
        if common_seq_len + prior_edit_seq_len > 512 - 3: # start of sequence token, end of sequence token and </prior_edits> token
            break
    source_seq += "</prior_edits>"
    
    # if args.debug:
    #     print(f"source_seq: \n{source_seq}")
    
    encoded_source_seq = tokenizer(source_seq, padding="max_length", truncation=True, max_length=512)
    source_ids = torch.tensor([encoded_source_seq["input_ids"]], dtype=torch.long)
    data = TensorDataset(source_ids)
    return data    