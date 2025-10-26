import sys
import json
import platform

sys.path.append("../")
from arg_val import *
from tqdm import tqdm
from code_window import CodeWindow
from clone_detect import find_similar_code_segment

def parse_identifier(code: bytes, lang):
    def traverse_tree(node, results, lang):
        if len(node.children) == 0 and "identifier" in node.type:
            results.append(node.text.decode("utf-8"))
        for child in node.children:
            traverse_tree(child, results, lang)
            
        return results
    
    system = platform.system().lower()
    if system == "darwin":
        build_dir = "../../dataset_collection/tree-sitter/macos_build"
    elif system == "linux":
        build_dir = "../../dataset_collection/tree-sitter/linux_build"
    elif system == "windows":
        build_dir = "../../dataset_collection/tree-sitter/windows_build"
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

    so_path = os.path.join(build_dir, "my-languages.so")
    
    LANGUAGE = Language(so_path, lang)

    parser = Parser()
    parser.set_language(LANGUAGE)
    
    tree = parser.parse(code)
    root_node = tree.root_node

    return traverse_tree(root_node, [], lang)

def is_rename_edit(old_identifiers, new_identifiers):
    if len(old_identifiers) != len(new_identifiers):
        # if the number of identifiers is different, it must be a rename edit
        return False, None, None
    
    old_identifiers = set(old_identifiers)
    new_identifiers = set(new_identifiers)
    if len(old_identifiers) != len(new_identifiers):
        # if the number of identifiers set is different, it must be a rename edit
        return False, None, None
    
    old_new_diff = list(old_identifiers.difference(new_identifiers))
    new_old_diff = list(new_identifiers.difference(old_identifiers))
    if len(old_new_diff) != len(new_old_diff):
        return False, None, None
    
    if len(old_new_diff) == 0:
        return False, None, None
    
    return True, old_new_diff, new_old_diff

def main(file_path):
    """
    Func:
        Analyze what kind of external tool message will be fed into generator
        The message including: 
            1. code clone
            2. reference edit
            3. definition edit
            4. rename edit
            5. normal
    Args:
        file_path: str, the path of the json file
    Return:
        None
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    total_cnt = 0
    with_external_tool_feedback = {
        "code clone": 0,
        "reference edit": 0,
        "definition edit": 0,
        "rename edit": 0
    }
    for commit_url, commit in tqdm(data.items()):
        lang = commit["lang"]
        for hunk_idx, raw_hunk in enumerate(commit["hunks"]):
            if "external_tool_feedback" in raw_hunk:
                # remove key
                del raw_hunk["external_tool_feedback"]
            total_cnt += 1
            hunk = CodeWindow(raw_hunk, "hunk")
            # check if this hunk is a reference edit or definition edit
            before_edit_region = hunk.before_edit_region(split_by_line=False, allow_fuzzy=False)
            after_edit_region = hunk.after_edit_region(split_by_line=False)
            
            before_arg_cases = parse_code(before_edit_region.encode(), lang)
            after_arg_cases = parse_code(after_edit_region.encode(), lang)
            
            diff_result = arg_case_diff(before_arg_cases, after_arg_cases)
            if diff_result:
                if diff_result["type"] == "ref":
                    raw_hunk["external_tool_feedback"] = "definition edit"
                    with_external_tool_feedback["definition edit"] += 1
                elif diff_result["type"] == "def":
                    raw_hunk["external_tool_feedback"] = "reference edit"
                    with_external_tool_feedback["reference edit"] += 1
                continue
            
            # check if this hunk is a rename edit
            old_identifiers = parse_identifier(before_edit_region.encode(), lang)
            new_identifiers = parse_identifier(after_edit_region.encode(), lang)
            
            is_rename, old_new_identifiers, new_old_identifiers = is_rename_edit(old_identifiers, new_identifiers)
            if is_rename:
                if find_similar_code_segment(before_edit_region, after_edit_region, threshold=90) and find_similar_code_segment(after_edit_region, before_edit_region, threshold=90):
                    if len(old_new_identifiers) == 0 or len(new_old_identifiers) == 0:
                        raise ValueError("The identifiers should not be empty")
                    raw_hunk["external_tool_feedback"] = f"rename edit, {old_new_identifiers} -> {new_old_identifiers}"
                    with_external_tool_feedback["rename edit"] += 1
                    continue
            
            # check if this hunk is a code clone
            other_hunks = [commit["hunks"][i] for i in range(len(commit["hunks"])) if i != hunk_idx]
            if before_edit_region == "" and "insert" in hunk.inter_labels:
                insert_pos = hunk.inter_labels.index("insert")
                query = hunk.code_window[insert_pos-1]
            else:
                query = before_edit_region
            for other_hunk in other_hunks:
                other_hunk_code = CodeWindow(other_hunk, "hunk")
                other_hunk_code = other_hunk_code.before_edit_region(split_by_line=False) # allow fuzzy because we are searching for similar code 
                if other_hunk_code == "" and "insert" in other_hunk["inter_labels"]:
                    insert_pos = other_hunk["inter_labels"].index("insert")
                    document = other_hunk["code_window"][insert_pos-1]
                else:
                    document = other_hunk_code
                
                # print(f"Query: {query}")
                # print(f"Document: {document}")
                found_segments = find_similar_code_segment(query, document, threshold=90)
                if found_segments:
                    raw_hunk["external_tool_feedback"] = "code clone"
                    with_external_tool_feedback["code clone"] += 1
                    break
            
            if "external_tool_feedback" not in raw_hunk:
                raw_hunk["external_tool_feedback"] = "normal"
            
            
    print(f"Total hunk count: {total_cnt}")
    print(f"Hunk count with external tool feedback: {with_external_tool_feedback}")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
                
if __name__ == "__main__":
    for mode in ["train", "dev", "test"]:
        main(f"../../dataset/all/{mode}.json")