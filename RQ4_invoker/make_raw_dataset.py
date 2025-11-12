import os
import sys
sys.path.append("../RQ5_simulation")
import json
import random
from tqdm import tqdm
from commit import Commit
from ask_lsp import ask_lsp, start_lsp
from logic_gate import get_edit_type_in_batch

random.seed(42)

def main(commit_url, lang, samples):
    commit = Commit(commit_url)
    commit.language = lang
    
    # get the possible edit type for each edit
    edits = []
    for edit_idx in range(commit.hunk_num()):
        edit = commit.get_edit(edit_idx)
        edits.append(edit)
        
    # we dont need to assign the edits back to commit object, 
    # because the extra information is stored directly in the commit object
    get_edit_type_in_batch(edits, commit.language)
    
    files_to_change = [os.path.normpath(os.path.join(commit.project_dir, file_path)) for file_path in commit.changed_files]
    LSP = start_lsp(commit.language, files_to_change, commit.project_dir)
    for target_edit in edits:
        if target_edit["edit_type"] == "normal":
            continue
        
        if len(target_edit["edit_info"]["propagatable_edit_idx"]) == 1:
            continue
        
        commit.prev_edits = []
        commit.prev_edits.append(target_edit)
        target_edit["simulated"] = True
        
        # Feed this version to LSP
        
        try:
            reported_lines = ask_lsp(commit, LSP)
        except:
            continue
        TP_cnt, FP_cnt = match_locations(reported_lines, commit)
        
        if TP_cnt == 0 and FP_cnt == 0:
            continue
        
        precision = TP_cnt / (TP_cnt + FP_cnt)
        
        if target_edit["edit_type"] == "rename":
            if target_edit["edit_info"]["deleted_identifiers"][0]["identifier_type"] == "function":
                class_type = "function_rename"
            elif target_edit["edit_info"]["deleted_identifiers"][0]["identifier_type"] == "variable":
                class_type = "variable_rename"
        else:
            class_type = target_edit["edit_type"]
        sample = {
            "commit_url": commit_url,
            "language": lang,
            "class": class_type,
            "target_hunk": target_edit,
            "TP_cnt": TP_cnt,
            "FP_cnt": FP_cnt
        }

        samples.append(sample)
        
        target_edit["simulated"] = False

    try:
        LSP.close()
    except:
        pass
    
    return samples
        
def match_locations(reported_lines, commit):
    if reported_lines is None:
        return 0, 0
    edit_ranges = {}
    for rel_file_path, snapshot in commit.snapshots.items():
        abs_file_path = os.path.normpath(os.path.join(commit.project_dir, rel_file_path))
        line_idx = 0
        for window in snapshot:
            if isinstance(window, list):
                line_idx += len(window)
                continue
            if window["simulated"] and window["edit_type"] != "rename":
                line_idx += len(window["after"])
            elif window["simulated"] and window["edit_type"] == "rename":
                line_idx += len(window["before"])
            else:
                if abs_file_path not in edit_ranges:
                    edit_ranges[abs_file_path] = []
                if len(window["before"]) > 0:
                    edit_ranges[abs_file_path].append([i for i in range(line_idx, line_idx + len(window["before"]))])
                line_idx += len(window["before"])
                
    TP_cnt, FP_cnt = 0, 0
    location_cnt = 0
    for abs_file_path, reported_locations in reported_lines.items():
        if abs_file_path not in edit_ranges:
            FP_cnt += len(reported_locations)
        else:
            for reported_location in reported_locations: # given a reported location
                for edit_range in edit_ranges[abs_file_path]: # match with all possible edit ranges
                    if reported_location["range"]["start"]["line"] in edit_range:
                        TP_cnt += 1
                        break
                else:
                    FP_cnt += 1
                
        location_cnt += len(reported_locations)
    
    try:
        assert TP_cnt + FP_cnt == location_cnt
    except:
        raise ValueError(f"TP_cnt ({TP_cnt}) + FP_cnt ({FP_cnt}) != location_cnt ({location_cnt})")
    return TP_cnt, FP_cnt

def perimitive_check(commit_url, lang):
    try:
        commit = Commit(commit_url)
        commit.language = lang
    except:
        print(f"{commit_url}: Cannot parse")
        return False
    
    # get the possible edit type for each edit
    edits = []
    for edit_idx in range(commit.hunk_num()):
        edit = commit.get_edit(edit_idx)
        edits.append(edit)
        
    # we dont need to assign the edits back to commit object, 
    # because the extra information is stored directly in the commit object
    get_edit_type_in_batch(edits, commit.language)
    
    # check if contain edits that are propagatable
    propagatable = False
    propagatable_edits = []
    for edit in edits:
        if edit["edit_type"] != "normal" and len(edit["edit_info"]["propagatable_edit_idx"]) > 1:
            propagatable = True
            propagatable_edits.append(edit["edit_type"])
    if not propagatable:
        print(f"{commit_url}: Not propagatable")
        return False
    else:
        print(f"{commit_url}: Propagatable, {set(propagatable_edits)}")
    
    if list(set(propagatable_edits)) == ["clone"]:
        # Too many clone edits, only 1/4 will return True
        return random.random() < 0.25 and propagatable
    
    return propagatable
 
def run_primitive_check(dataset_type):
    """
    Run primitive check. Use AST and other static method to filter out commit that definitely do not contain our pre-defined edits.
    
    """
    with open(f"../dataset/all/{dataset_type}.json", "r") as f:
        dataset = json.load(f)
    
    if not os.path.exists(f"potential/{dataset_type}_potential_commits.json"):
        potential_commits = []
    else:
        with open(f"potential/{dataset_type}_potential_commits.json", "r") as f:
            potential_commits = json.load(f)
    
    urls = list(dataset.keys())
    
    # Resume from last processed commit
    if len(potential_commits) > 0:
        last_potential_commit = potential_commits[-1]
        last_potential_commit_idx = urls.index(last_potential_commit)
        urls = urls[last_potential_commit_idx + 1:]
    
    for idx, commit_url in enumerate(tqdm(urls)):
        if commit_url in potential_commits:
            continue
        
        commit = dataset[commit_url]
        language = commit["lang"]

        if perimitive_check(commit_url, language):
            potential_commits.append(commit_url)

            with open(f"potential/{dataset_type}_potential_commits.json", "w") as f:
                json.dump(potential_commits, f, indent=4)

if __name__ == "__main__":
    os.makedirs("potential", exist_ok=True) # this folder for commits that contain edits that are poentially our pre-defined edits
    os.makedirs("raw_dataset", exist_ok=True) # this folder for edits that can/cannot use lsp to detect propagation, not split into train/valid/test yet, not ready to feed into invoker
    
    for dataset_type in  ["train", "dev", "test"]:
        run_primitive_check(dataset_type)

        with open(f"potential/{dataset_type}_potential_commits.json", "r") as f:
            dataset = json.load(f)
                
        with open(f"raw_dataset/{dataset_type}.json", "w") as f:
            json.dump([], f, indent=4)
        
        with open(f"../dataset/all/{dataset_type}.json", "r") as f:
            original_dataset = json.load(f)
            
        last_proj = ""
        for commit_url in tqdm(dataset):
            curr_proj = commit_url.split("/")[-3]
            commit_info = original_dataset[commit_url]
            language = commit_info["lang"]
            if last_proj != curr_proj and last_proj != "":
                print(f"Switch to {curr_proj}")
            last_proj = curr_proj
            samples_in_commit = main(commit_url, language, [])
            # add samples_in_commit to meta_dataset.json
            with open(f"raw_dataset/{dataset_type}.json", "r") as f:
                meta_dataset = json.load(f)
            meta_dataset.extend(samples_in_commit)
            with open(f"raw_dataset/{dataset_type}.json", "w") as f:
                json.dump(meta_dataset, f, indent=4)
