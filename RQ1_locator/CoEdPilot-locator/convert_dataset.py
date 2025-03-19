# This script is used to convert new dataset to old dataset, to test performance of ISSTA locator
import os
import json

from code_window import *
from tqdm import tqdm
from rank_bm25 import BM25Okapi

def formalize_locator_input(sliding_window: CodeWindow, prompt: str, prior_edits: list[dict]) -> tuple[str, str]:
    source_seq = ""
    for line in sliding_window.code_window:
        source_seq += f"<mask>{line}"
    # prepare the prompt region
    # truncate prompt if it encode to more than 64 tokens
    source_seq += f"</s>{prompt}</s>"
    # prepare the prior edits region
    for prior_edit in prior_edits:
        # Allow the last prior edit to be truncated (Otherwise waste input spaces)
        source_seq += f"remove {prior_edit.before_edit_region(False)} </s> add {prior_edit.before_edit_region(False)} </s>"
        
    return source_seq

def label_conversion(inline_labels: list[str], inter_labels: list[str]) -> list[str]:
    """
    Func:
        Given the fine grain label of new models, convert them to old labels
    Args:   
        inline_labels: list[str], have label: keep, replace, delete
        inter_labels: list[str], have label: null, insert, block-split
    Return:
        old_labels: list[str]
    """
    assert len(inline_labels) + 1 == len(inter_labels)
    # rule 1: block-split can be ignored
    inter_labels = ["null" if x == "block-split" else x for x in inter_labels]
    
    # rule 2: delete is a part of replace
    inline_labels = ["replace" if x == "delete" else x for x in inline_labels]
    
    # rule 3: old labels can't handle insert at the beginning of code window
    inter_labels = inter_labels[1:]
    
    old_labels = []
    # rule 4: now inter_label  should only have null & insert
    #             inline_label should only have keep & replace
    for inter_label, inline_label in zip(inter_labels, inline_labels):
        if inter_label == "null":
            old_labels.append(inline_label)
        else: # inter_label == "insert"
            if inline_label == "keep":
                old_labels.append("add")
            else:
                old_labels.append("replace")
                
    assert len(old_labels) == len(inline_labels)
    return old_labels
                

def convert_dataset(dataset_path, lang):
    for name in ["train", "dev", "test"]:
        with open(os.path.join(dataset_path, lang, f"{name}.json"), "r") as f:
            raw_dataset = json.load(f)
        
        raw_locator_dataset = []
        for idx, (commit_url, commit) in enumerate(tqdm(raw_dataset.items(), desc="Finding relevant prior edits")): # for each commit
            commit_msg = commit["commit_msg"]
            hunks = [CodeWindow(h, "hunk") for h in commit["hunks"]]
            sliding_windows = commit["sliding_windows"]
            
            # form estimator dataset
            for sliding_window in sliding_windows:
                sliding_window = CodeWindow(sliding_window, "sliding_window")
                non_overlap_hunks = [hunk for hunk in hunks if hunk.id not in sliding_window.overlap_hunk_ids]
                choosen_hunk_ids = [hunk.id for hunk in hunks if hunk.id not in sliding_window.overlap_hunk_ids] # index to hunk id
                tokenized_corpus = ["".join(hunk.before_edit_window()+hunk.after_edit_region()).split() for hunk in non_overlap_hunks]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = "".join(sliding_window.code_window).split()
                retrieval_code = bm25.get_top_n(tokenized_query, tokenized_corpus, n=3) 
                retrieved_index = [tokenized_corpus.index(i) for i in retrieval_code] # get index in choosen_hunk_ids
                prior_edit_id = [choosen_hunk_ids[idx] for idx in retrieved_index] # get corresponding hunk id
                prior_edits = []
                for id in prior_edit_id: # preserve the order
                    prior_edits.append([hunk for hunk in hunks if hunk.id == id][0])
                # for picked prior edits, combine with sliding window, turn into locator data sample
                source_seq = formalize_locator_input(sliding_window, commit_msg, prior_edits)
                raw_locator_dataset.append({
                    "code_tokens": source_seq,
                    "docstring_tokens": " ".join(label_conversion(sliding_window.inline_labels, sliding_window.inter_labels)),
                    "commit_url": commit_url
                })
        os.makedirs(f"./dataset_from_fine_grain/{lang}", exist_ok=True)
        with open(f"./dataset_from_fine_grain/{lang}/{name}.jsonl", "w") as f:
            for data in raw_locator_dataset:
                f.write(json.dumps(data) + "\n")
    
    return raw_locator_dataset
        

if __name__ == "__main__":
    dataset_path = "/media/user/dataset_fine_grain"
    lang = "all"
    convert_dataset(dataset_path, lang)