import os
import json
import tree_sitter
from tree_sitter import Language, Parser
from sklearn.metrics import classification_report

def parse(code: str, language: str):
    assert language in ["go", "javascript", "typescript", "python", "java"]
    if not os.path.exists("/home/user/workspace/CodeEdit/locator-fine-grain/metric/tree-sitter/build/my-languages.so"):
        Language.build_library(
            # Store the library in the `build` directory
            "/home/user/workspace/CodeEdit/locator-fine-grain/metric/tree-sitter/build/my-languages.so",

            # Include one or more languages
            [
                "metric/tree-sitter/tree-sitter-go",
                "metric/tree-sitter/tree-sitter-javascript",
                "metric/tree-sitter/tree-sitter-typescript/typescript",
                "metric/tree-sitter/tree-sitter-python",
                "metric/tree-sitter/tree-sitter-java",
            ]
        )
    parser = Parser()
    parser.set_language(Language("/home/user/workspace/CodeEdit/locator-fine-grain/metric/tree-sitter/build/my-languages.so", language))
    tree = parser.parse(bytes(code, "utf8"))
    return tree

def get_variable_accesses(code, language):
    tree = parse(code, language)
    root_node = tree.root_node

    reads, writes = set(), set()

    def traverse(node):
        if node.type == 'identifier':
            parent = node.parent
            if parent.type in ['assignment', 'augmented_assignment']:
                # 左侧是写操作，右侧是读操作
                left_node = parent.child_by_field_name('left')
                right_node = parent.child_by_field_name('right')
                if node == left_node:
                    writes.add(node.text.decode('utf-8'))
                elif node == right_node:
                    reads.add(node.text.decode('utf-8'))
            else:
                reads.add(node.text.decode('utf-8'))
        
        # 遍历子节点
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return reads, writes

def analyze_data_dependencies(code1, code2, language):
    reads1, writes1 = get_variable_accesses(code1, language)
    reads2, writes2 = get_variable_accesses(code2, language)

    write_read = writes1 & reads2
    write_write = writes1 & writes2
    read_write = reads1 & writes2

    return bool(write_read or write_write or read_write)

def get_min_level_per_line(node: tree_sitter.Node, level=0, line_levels=None):
    """
    Func:
        Given an AST node, we label the level of each line of code
    Args:
        node: AST node
        level: the level of code
        line_levels: dict, records level of each line
    Return:
        line_levels: dict
    """
    def merge_dicts_with_min(*dicts):
        result = {}
        for d in dicts:
            for key, value in d.items():
                if key in result:
                    result[key] = min(result[key], value)
                else:
                    result[key] = value
        return result
    if line_levels is None:
        line_levels = {}

    start_line = node.start_point[0]
    end_line = node.end_point[0]
    if len(node.children) == 0:
        for line in range(start_line, end_line+1):
            if line not in line_levels or level < line_levels[line]:
                line_levels[line] = level
    else:
        for child in node.children:
            line_levels = merge_dicts_with_min(line_levels, get_min_level_per_line(child, level + 1, line_levels))

    return line_levels

def find_adjacent_same_numbers(nums, min_consecutive_cnt):
    if not nums:
        return []

    result = []
    current_num = nums[0]
    start_index = 0
    length = 1

    for i in range(1, len(nums)):
        if nums[i] == current_num:
            length += 1
        else:
            if length >= min_consecutive_cnt:
                result.append((current_num, start_index, start_index + length - 1))
            current_num = nums[i]
            start_index = i
            length = 1

    # check last paragraph
    if length >= min_consecutive_cnt:
        result.append((current_num, start_index, start_index + length - 1))

    return result

def find_equivalent_location(code_window: list[str], to_insert_code: str, language: str):
    """
    Func:
        Given the code window and code to insert, find inter-line locations that are equivalent
    Args:
        code_window: list[str], lines of code
        to_insert_code: str, code to insert
        language: str
    Return:
        filtered_positions: list of list, each sublist is equvialent locations
    """
    assert language in ["go", "javascript", "typescript", "python", "java"]
    code_window_str = "".join(code_window)
    code_window_tree = parse(code_window_str, language)
    line_level_dict = get_min_level_per_line(code_window_tree.root_node)
    line_levels = [-1 for i in range(len(code_window))]
    for line_idx, level in line_level_dict.items():
        if line_idx < len(line_levels):
            line_levels[line_idx] = level
    
    # check each line's dependency relationship with code to insert
    for line_idx, line in enumerate(code_window):
        if analyze_data_dependencies(line, to_insert_code, language): 
            # if exist dependency relationship, we disrupt level to avoid adjacency
            line_levels[line_idx] = max(line_levels) + 1
          
    positions = find_adjacent_same_numbers(line_levels, 2)
    filtered_positions = []
    for p in positions:
        if p[1] == 0 and p[2] == len(line_levels)-1: 
            eqv_loc = [i for i in range(p[1], p[2]+2)]
        elif p[2] - p[1] >= 2 and p[1] != 0 and p[2] != len(line_levels)-1: 
            eqv_loc = [i for i in range(p[1]+1, p[2]+1)]
        elif p[2] - p[1] >= 2 and p[1] == 0: 
            eqv_loc = [i for i in range(p[1], p[2]+1)]
        elif p[2] - p[1] >= 2 and p[2] == len(line_levels)-1: 
            eqv_loc = [i for i in range(p[1]+1, p[2]+2)]
        elif p[2] - p[1] == 1 and line_levels[p[1]] == -1:
            eqv_loc = [i for i in range(p[1], p[2]+2)]
        elif p[2] - p[1] == 1 and p[1] == 0: 
            eqv_loc = [i for i in range(p[1], p[2]+1)]
        elif p[2] - p[1] == 1 and p[2] == len(line_levels)-1: 
            eqv_loc = [i for i in range(p[1]+1, p[2]+2)]
        else:
            continue
        filtered_positions.append(eqv_loc)

    return filtered_positions

def merge_overlapping_lists(list_of_lists):
    merged = True
    
    while merged:
        merged = False
        new_list_of_lists = []
        while list_of_lists:
            # get first list
            first, *rest = list_of_lists
            first_set = set(first)
            
            # check if the rest list has overlap 
            rest2 = []
            for l in rest:
                if first_set & set(l):  # if there is overlap 
                    first_set |= set(l)  # merge
                    merged = True
                else:
                    rest2.append(l)
            
            new_list_of_lists.append(list(first_set))
            list_of_lists = rest2
        
        list_of_lists = new_list_of_lists
    
    # sort each sub-list 
    for i in range(len(list_of_lists)):
        list_of_lists[i].sort()
        
    # rank list based on their first element
    list_of_lists.sort(key=lambda x: x[0] if x else float('inf'))
    
    return list_of_lists

def insert_remaining_as_lists(merged_list_of_lists, actual_length):
    # extract all ints that are merged 
    merged_elements = set()
    for lst in merged_list_of_lists:
        merged_elements.update(lst)
    
    # find rest of int and insert them as single element list 
    remaining_elements = set(range(actual_length)) - merged_elements
    for elem in remaining_elements:
        merged_list_of_lists.append([elem])
    
    # sort based on final list
    merged_list_of_lists.sort(key=lambda x: x[0])
    
    return merged_list_of_lists
  
def fuzzy_metric(sliding_window: dict, inter_line_preds: list[str], language):
    """
    Args:
        sliding_window: dict_keys(['code_window', 'inline_labels', 'inter_labels', 
            'overlap_hunk_ids', 'file_path', 'edit_start_line_idx', 
            'sliding_window_type', 'to_insert'])
        inter_line_preds: list of predictions for each line, do not have <>
    """
    all_equivalent_locs = []
    for to_insert_code in sliding_window["to_insert"]:
        equivalent_locations = find_equivalent_location(
                                    sliding_window["code_window"],
                                    "".join(to_insert_code), language)
        all_equivalent_locs.extend(equivalent_locations)
        
    all_equivalent_locs = merge_overlapping_lists(all_equivalent_locs)
    all_equivalent_locs = insert_remaining_as_lists(all_equivalent_locs, len(inter_line_preds))
    combined_preds = []
    combined_golds = []

    for eqv_loc in all_equivalent_locs:
        pred = [inter_line_preds[i] for i in eqv_loc]
        gold = [sliding_window["inter_labels"][i] for i in eqv_loc]
        if "insert" in pred:
            combined_preds.append("insert")
        else:
            combined_preds.append("null")
            
        if "insert" in gold:
            combined_golds.append("insert")
        else:
            combined_golds.append("null")
    
    return combined_preds, combined_golds
    
def analyze_prediction(model_path, dataset_path, lang):
    with open(os.path.join(dataset_path, lang, "test.json"), "r") as f:
        test_data = json.load(f)
    
    with open(os.path.join(model_path, lang, "test_bm25.pred"), "r") as f:
        preds = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
        
    locator_dataset = []
    dataset_lang = []
    for _, commit in test_data.items():
        locator_dataset.extend(commit["sliding_windows"])
        if lang == "all":
            dataset_lang.extend([commit["lang"]]*len(commit["sliding_windows"]))
    
    try:
        assert len(locator_dataset) == len(preds)
        assert len(dataset_lang) == len(locator_dataset)
    except AssertionError:
        print(f"Error: locator dataset size {len(locator_dataset)} != preds size {len(preds)}")
    
    inter_line_preds = []
    for pred in preds:
        inter_line_pred = []
        for i, p in enumerate(pred):
            if i % 2 == 0:
                inter_line_pred.append(p[1:-1]) # remove "<" and ">"
        inter_line_preds.append(inter_line_pred)
    
    fuzzy_all_preds = []
    fuzzy_all_golds = []
    all_preds = []
    all_golds = []
    for sliding_window, preds, lang in zip(locator_dataset, inter_line_preds, dataset_lang):
        combined_preds, combined_golds = fuzzy_metric(sliding_window, preds, lang)
        all_preds.extend(preds)
        all_golds.extend(sliding_window["inter_labels"])
        fuzzy_all_preds.extend(combined_preds)
        fuzzy_all_golds.extend(combined_golds)
    
    fuzzy_report = classification_report(fuzzy_all_golds, fuzzy_all_preds, output_dict=True)
    original_report = classification_report(all_golds, all_preds, output_dict=True)
    return fuzzy_report["insert"], original_report["insert"]
    
if __name__ == "__main__":
    fuzzy_metric_score, original_metric_score = analyze_prediction("model-codet5-base", "/media/user/dataset_fine_grain", "python")
    print("\t\tPrecision\tRecall\t\tF1\t\tSupport")
    print(f"Fuzzy\t\t{fuzzy_metric_score['precision']:.4f}\t\t{fuzzy_metric_score['recall']:.4f}\t\t{fuzzy_metric_score['f1-score']:.4f}\t\t{fuzzy_metric_score['support']}")
    print(f"Original\t{original_metric_score['precision']:.4f}\t\t{original_metric_score['recall']:.4f}\t\t{original_metric_score['f1-score']:.4f}\t\t{original_metric_score['support']}")