import os
import re
import subprocess

from tree_sitter import Language, Parser

def clone_repo(user_name: str, project_name: str):
    """
    Clone the repository to local
    
    Args:
        user_name: str, the user name of the repository
        project_name: str, the name of the repository
        
    Returns:
        None
    """
    command = f"git clone https://github.com/{user_name}/{project_name}.git /media/user/repos/{project_name}"
    subprocess.run(command, shell=True)

def detect_extension(file_names: list[str]):
    # Use os.path.basename to get file name
    for file_name in file_names:
        filename = os.path.basename(file_name)
        # Use splitext to split file name and extension
        file_name_elements = filename.split('.')
        if len(file_name_elements) == 2:
            extension = '.'+file_name_elements[-1]
        else:
            extension =  '.'+'.'.join(file_name_elements[-2:])
        white_list = ['.go', '.js', '.java', '.py', '.ts', '.tsx']
        if extension not in white_list:
            return True
    return False
    
def extract_hunks(commit_url: str):
    """
    Given commit url, extract edit hunks from the commit, with its file path and code logic path
    
    Args:
        commit_url: str, the url of the commit
        
    Returns:
        commit_message: str, the message of the commit
        commit_snapshots: dict, key is file path, value is list of snapshot of the file
    """
    commit_sha = commit_url.split("/")[-1]
    project_name = commit_url.split("/")[-3]
    user_name = commit_url.split("/")[-4]
    repo_path = os.path.join("/media/user/repos", project_name)
    
    # if not exist, clone to local
    if not os.path.exists("/media/user/repos"):
        os.mkdir("/media/user/repos")
    if not os.path.exists(repo_path):
        clone_repo(user_name, project_name)
    
    command = f"git -C {repo_path} checkout -f {commit_sha}^"
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except:
        raise ValueError(f'3 {commit_url} Error: Error in git checkout')
    
    command = f"git -C {repo_path} show {commit_sha} --pretty=%B --no-patch"
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except:
        raise ValueError(f'1 {commit_url} Error: Error in retrieving commit message')
    commit_message = result.stdout.strip()
    
    command = f"git -C {repo_path} checkout {commit_sha}^"
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except:
        raise ValueError(f'2 {commit_url} Error: Error in git checkout')
    
    command = f'git -C {repo_path} diff -U10000000 {commit_sha}^ {commit_sha}'
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except:
        raise ValueError(f'1 {commit_url} Error: Error in git diff')
    git_diff_str = result.stdout
    
    file_name_matches = re.finditer(r'diff --git a/(.+) b/(.+)', git_diff_str)
    file_names = []
    for match in file_name_matches:
        before_filename = match.group(1)
        after_filename = match.group(2)
        try:
            assert before_filename == after_filename
        except:
            raise ValueError(f"{commit_url} Error: Contain edit changes file name: {before_filename} -> {after_filename}")
        file_names.append(before_filename)
    
    if detect_extension(file_names):
        raise ValueError(f'{commit_url} Error: Contain edit on non-source files')
    
    # Split into diff section, 1 section = 1 file
    diff_sections = re.findall(r'diff --git[^\n]*\n.*?(?=\ndiff --git|$)', git_diff_str, re.DOTALL)
    all_edit_num = 0
    commit_snapshots = {}
    for i, section in enumerate(diff_sections):
        # Parse file name (w/ path), make sure edit don't change file name
        file_name_match = re.match(r'diff --git a/(.+) b/(.+)', section)
        if file_name_match:
            file_name = file_name_match.group(1)
        else:
            raise ValueError(f"5 {commit_url} Error: file name contain non-ascii char")
        
        # Get the diff of the whole file
        # (if -U{number} is set large enough, a file should contain only 1 @@ -xx,xx +xx,xx @@)
        # we can only make snapshot based on the diff of the whole file
        match = re.search(r'@@[^\n]*\n(.+)', section, re.DOTALL)
        if not match:
            raise ValueError(f"4 {commit_url} Error: Edit fail to match @@ -xx,xx +xx,xx @@")
        # Match content after line @@
        after_at_symbol_content = match.group(1)
        # form snapshot: each element:
        # type 1: list of line of code, unchanged
        # type 2: dict of edit, have key: "idx", "type", "before", "after", "simulated"
        snapshot, _ = convert_diff_section_to_snapshot(after_at_symbol_content)
        
        commit_snapshots[file_name] = snapshot
        
    # extract code logic path for each hunk
    hunk_idx = 0
    for file_path, snapshot in commit_snapshots.items():
        file_path = os.path.join(repo_path, file_path)
        for window in snapshot:
            if type(window) is list:
                continue
            
            window["idx"] = hunk_idx
            hunk_idx += 1
            
    return commit_message, commit_snapshots

def convert_diff_section_to_snapshot(file_w_diff: str):
    """
    Func:
        from "git -diff ..." output to snapshot
    """
    diff_content = file_w_diff.splitlines(keepends=True)
    snapshot = []
    consecutive_code = []
    under_edit = False
    edits = []
    for line in diff_content:
        if line.startswith(" ") and under_edit == False:
            consecutive_code.append(line[1:])
        elif line.startswith(" ") and under_edit == True:
            under_edit = False
            if edit["type"] == "replace" and edit["after"] == []:
                edit["type"] = "delete"
            snapshot.append(edit.copy())
            consecutive_code.append(line[1:]) 
        elif line.startswith("-") and under_edit == False:
            under_edit = True
            if consecutive_code != []:
                snapshot.append(consecutive_code.copy())
            consecutive_code = []
            edit = {
                "type": "replace",
                "before": [],
                "after": [],
                "simulated": False
            }
            edit["before"].append(line[1:])
        elif line.startswith("+") and under_edit == False:
            under_edit = True
            if consecutive_code != []:
                snapshot.append(consecutive_code.copy())
            consecutive_code = []
            edit = {
                "type": "insert",
                "before": [],
                "after": [],
                "simulated": False
            }
            edit["after"].append(line[1:])
        elif line.startswith("+") and under_edit == True:
            edit["after"].append(line[1:])
        elif line.startswith("-") and under_edit == True:
            edit["before"].append(line[1:])
    if under_edit == True:
        if edit["type"] == "replace" and edit["after"] == []:
            edit["type"] = "delete"
        snapshot.append(edit.copy())
    if under_edit == False:
        snapshot.append(consecutive_code.copy())
    
    for window in snapshot:
        if type(window) == dict:
            edits.append(window)
    return snapshot, edits

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
                old_labels.append("insert")
            else:
                old_labels.append("replace")
                
    assert len(old_labels) == len(inline_labels)
    return old_labels

def label_conversion_reverse(labels: list[str]):
    inline_labels = labels
    inter_labels = ["null"] * (len(inline_labels) + 1)
    for i, label in enumerate(inline_labels):
        if label == "insert":
            inline_labels[i] = "keep"
            inter_labels[i+1] = "insert"
            
    return inline_labels, inter_labels

def diagnostic_2_sliding_windows(diagnostics, commit):
    """
    Func:
        Convert lsp diagnostics to sliding windows
    Input:
        diagnostics: list of dict:
            {
                "source": "pylint",
                "range": {
                    "start": {
                        "line": 21,
                        "character": 0
                    },
                    "end": {
                        "line": 21,
                        "character": 10
                    }
                },
                "message": "[unused-import] Unused import re",
                "severity": 2,
                "code": "W0611",
                "tags": [1],
                "file_path": "airflow/providers/amazon/aws/hooks/sagemaker.py"
            }
        commit: Commit
    """
    sliding_windows = []
    for diagnostic in diagnostics:
        sliding_window = {}
        absolute_file_path = os.path.join(commit.project_dir, diagnostic["file_path"])
        with open(absolute_file_path, "r") as f:
            file_content = f.readlines()
            
        start_line_idx = max(0, diagnostic["range"]["start"]["line"] - 3)
        end_line_idx = min(len(file_content), diagnostic["range"]["end"]["line"] + 5)
        
        assert len(file_content) >= end_line_idx > start_line_idx
        
        sliding_window["code_window"] = file_content[start_line_idx:end_line_idx]
        sliding_window["file_path"] = diagnostic["file_path"]
        sliding_window["start_line_idx"] = start_line_idx
        sliding_window["file_lines"] = len(file_content)
        sliding_windows.append(sliding_window)
    
    return sliding_windows

def merge_predictions(A_predictions, B_predictions):
    """
    Func:
        Merge two predictions
    """
    if A_predictions is None or A_predictions == {}:
        return B_predictions
    if B_predictions is None or B_predictions == {}:
        return A_predictions
    
    merged_predictions = {}
    for file_path, A_prediction in A_predictions.items():
        merged_prediction = {
            "inline_predictions": [],
            "inline_confidences": [],
            "inter_predictions": [],
            "inter_confidences": [],
            "inline_service": [],
            "inter_service": []
        }
        B_prediction = B_predictions[file_path]
        A_inline_predictions = A_prediction["inline_predictions"]
        A_inline_confidences = A_prediction["inline_confidences"]
        A_inter_predictions = A_prediction["inter_predictions"]
        A_inter_confidences = A_prediction["inter_confidences"]
        A_inline_service = A_prediction["inline_service"]
        A_inter_service = A_prediction["inter_service"]
        
        B_inline_predictions = B_prediction["inline_predictions"]
        B_inline_confidences = B_prediction["inline_confidences"]
        B_inter_predictions = B_prediction["inter_predictions"]
        B_inter_confidences = B_prediction["inter_confidences"]
        B_inline_service = B_prediction["inline_service"]
        B_inter_service = B_prediction["inter_service"]
        
        # merge inline predictions
        for A_pred, A_conf, A_service, B_pred, B_conf, B_service in zip(A_inline_predictions, A_inline_confidences, A_inline_service, B_inline_predictions, B_inline_confidences, B_inline_service):
            # if one is actionable label and one is not, preserve the actionable one
            if A_pred in ["<replace>", "<delete>"] and B_pred == "<keep>":
                merged_prediction["inline_predictions"].append(A_pred)
                merged_prediction["inline_confidences"].append(A_conf)
                merged_prediction["inline_service"].append(A_service)
            elif B_pred in ["<replace>", "<delete>"] and A_pred == "<keep>":
                merged_prediction["inline_predictions"].append(B_pred)
                merged_prediction["inline_confidences"].append(B_conf)
                merged_prediction["inline_service"].append(B_service)
            elif A_conf > B_conf: 
                # if they are all actionable, or all not actionable, let the higher confidence one win
                merged_prediction["inline_predictions"].append(A_pred)
                merged_prediction["inline_confidences"].append(A_conf)
                merged_prediction["inline_service"].append(A_service)
            else:
                merged_prediction["inline_predictions"].append(B_pred)
                merged_prediction["inline_confidences"].append(B_conf)
                merged_prediction["inline_service"].append(B_service)
        
        # merge inter predictions
        for A_pred, A_conf, A_service, B_pred, B_conf, B_service in zip(A_inter_predictions, A_inter_confidences, A_inter_service, B_inter_predictions, B_inter_confidences, B_inter_service):
            if A_pred in ["<insert>", "<block-split>"] and B_pred == "<null>":
                merged_prediction["inter_predictions"].append(A_pred)
                merged_prediction["inter_confidences"].append(A_conf)
                merged_prediction["inter_service"].append(A_service)
            elif B_pred in ["<insert>", "<block-split>"] and A_pred == "<null>":
                merged_prediction["inter_predictions"].append(B_pred)
                merged_prediction["inter_confidences"].append(B_conf)
                merged_prediction["inter_service"].append(B_service)
            elif A_conf > B_conf:
                merged_prediction["inter_predictions"].append(A_pred)
                merged_prediction["inter_confidences"].append(A_conf)
                merged_prediction["inter_service"].append(A_service)
            else:
                merged_prediction["inter_predictions"].append(B_pred)
                merged_prediction["inter_confidences"].append(B_conf)
                merged_prediction["inter_service"].append(B_service)
        
        merged_predictions[file_path] = merged_prediction
        
    return merged_predictions
        
        