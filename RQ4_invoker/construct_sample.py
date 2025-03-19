import os
import sys
import json
import random
sys.path.append("../new_RQ4")

from tqdm import tqdm
from commit import Commit
from simhash import Simhash

def formalize_hunk(commit, hunk_idx):
    edit_hunk = commit.get_edit(hunk_idx)
    
    before_code = edit_hunk["before"]
    after_code = edit_hunk["after"]
     
    hunk_str = f"<before>{''.join(before_code)}</before><after>{''.join(after_code)}</after>"
    return hunk_str

def main(raw_dataset, language):
    dataset = []
    for raw_sample in tqdm(raw_dataset):
        commit = Commit(raw_sample["commit_url"])
        commit.language = language

        propagatable_edit_idx = raw_sample["target_hunk"]["edit_info"]["propagatable_edit_idx"]
        propagatable_edit_idx = list(set(propagatable_edit_idx) - {raw_sample["target_hunk"]["idx"]})
        target_edit_idx = raw_sample["target_hunk"]["idx"]
         
        target_edit_hunk_str = formalize_hunk(commit, target_edit_idx)
        propagatable_edit_hunk_strs = [formalize_hunk(commit, idx) for idx in propagatable_edit_idx]
        
        sample_input = f"<{raw_sample['class']}><last_edit>{target_edit_hunk_str}</last_edit>"
        if raw_sample['class'] == "clone" or random.random() < 0.25:
            for propagatable_edit_hunk_str in propagatable_edit_hunk_strs:
                sample_input += f"<previous_edit>{propagatable_edit_hunk_str}</previous_edit>"
        
        if raw_sample["TP_cnt"] == 0:
            if raw_sample["FP_cnt"] == 0:
                binary_label = 1
            else:
                binary_label = 0
        else:
            binary_label = 1
            
        sample = {
            "input": sample_input,
            "class": raw_sample['class'],
            "binary_label": binary_label,
            "regression_label": raw_sample["TP_cnt"] / (raw_sample["TP_cnt"] + raw_sample["FP_cnt"]),
            "commit_url": raw_sample["commit_url"],
            "target_edit_idx": target_edit_idx,
            "propagatable_edit_idx": propagatable_edit_idx
        }
        
        dataset.append(sample)
    
    return dataset

def deduplicate(language):
    def get_simhash(text):
        return Simhash(text).value

    with open(f"dataset/{language}.json", "r") as f:
        data = json.load(f)
        
    # read data
    unique_hashes = set()
    deduplicated_data = []

    for item in data:
        hash_value = get_simhash(item["input"])
        if hash_value not in unique_hashes:
            unique_hashes.add(hash_value)
            deduplicated_data.append(item)

    print(f"Before deduplication: {len(data)}, after deduplication: {len(deduplicated_data)}, preserved ratio: {len(deduplicated_data) / len(data)}")
    # save data after de-duplication
    with open(f"dataset/{language}.json", "w") as f:
        json.dump(deduplicated_data, f, indent=4)
    
if __name__ == "__main__":
    dataset_types = ["train", "dev", "test"]
    for dataset_type in dataset_types:
        print(f"Processing {dataset_type}...")
        with open(f"raw_dataset/{dataset_type}.json", "r") as f:
            raw_dataset = json.load(f)
            
        dataset = main(raw_dataset, dataset_type)
        
        
        os.makedirs("dataset", exist_ok=True)
        with open(f"dataset/{dataset_type}.json", "w") as f:
            json.dump(dataset, f, indent=4)
            
        deduplicate(dataset_type)
        
    