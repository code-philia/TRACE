# This script is used to convert snapshots within the same commit into a dataset
import os
import re
import json
import random
import itertools
import numpy as np
import transformers

from llama import *
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv("../.env")
ROOT_PATH = os.getenv("ROOT_PATH")
transformers.utils.logging.set_verbosity_error()

def filter_clean_msg_with_llama(msg: str, llama3, llama3_tokenizer):
    # construct prompt
    with open("prompt.txt", "r") as f:
        prompt = f.read()

    prompt = prompt.replace("<commit_message>", msg)
    # ask llama3
    b1 = b2 = None
    for i in range(4):
        answer = ask_llama3(prompt, llama3, llama3_tokenizer)
        b1, b2, cleaned_msg = parse_answer(answer)
        if b1 != None:
            break
    if b1 == False or b1 is None or b2 == False or b2 is None or len(cleaned_msg) == 0:
        raise ValueError("Low quality commit message")
    
    return cleaned_msg

def make_type3_sliding_window(windows, file_path):
    def merge_inter_labels(label1, label2):
        if len(label1) == 0:
            return label2
        if len(label2) == 0:
            return label1
        if label1[-1] == "insert" or label2[0] == "insert":
            return label1[:-1] + ["insert"] + label2[1:]
        else:
            return label1[:-1] + ["null"] + label2[1:]
        
    def combine_windows(window_triplet):
        big_window = {
            "code_window": [],
            "inline_labels": [],
            "inter_labels": [],
            "to_insert": [],
            "line_belong_to_hunk_id": []
        }
        for window in window_triplet:
            assert len(window["code_window"]) == len(window["inline_labels"])
            assert len(window["code_window"]) + 1 == len(window["inter_labels"])
            assert len(window["code_window"]) == len(window["line_belong_to_hunk_id"])
            big_window["code_window"].extend(window["code_window"])
            big_window["inline_labels"].extend(window["inline_labels"])
            big_window["inter_labels"] = merge_inter_labels(big_window["inter_labels"], window["inter_labels"])
            big_window["to_insert"].extend(window["to_insert"])
            big_window["line_belong_to_hunk_id"].extend(window["line_belong_to_hunk_id"])
        assert len(big_window["to_insert"]) == big_window["inter_labels"].count("insert")
        return big_window
    
    def split_window(big_window, window_size, file_path, target_hunk_id):
        small_windows = []
        insert_idx_base = 0
        for i in range(0, len(big_window["code_window"]), window_size):
            line_belong_to_hunk_id = big_window["line_belong_to_hunk_id"][i:i+window_size]
            overlap_hunk_ids = list(set(line_belong_to_hunk_id))
            if -1 in overlap_hunk_ids:
                overlap_hunk_ids.remove(-1)
            if overlap_hunk_ids is None:
                overlap_hunk_ids = []
            insert_label_cnt = big_window["inter_labels"][i:i+window_size+1].count("insert")
            small_window = {
                "code_window": big_window["code_window"][i:i+window_size],
                "inline_labels": big_window["inline_labels"][i:i+window_size],
                "inter_labels": big_window["inter_labels"][i:i+window_size+1],
                "overlap_hunk_ids": overlap_hunk_ids,
                "file_path": file_path,
                "to_insert": big_window["to_insert"][insert_idx_base:insert_idx_base+insert_label_cnt],
                "edit_start_line_idx": -1,
                "sliding_window_type": "type3",
                "previous_hunk_id": target_hunk_id
            }
            for idx in [0, -1]: # avoid block-split appearing anywhere beyond the middle of 2 replace
                if small_window["inter_labels"][idx] == "block-split":
                    small_window["inter_labels"][idx] = "null"
            insert_idx_base += insert_label_cnt
            if len(small_window["code_window"]) > 5:
                assert len(small_window["to_insert"]) == small_window["inter_labels"].count("insert")
                small_windows.append(small_window)
        return small_windows
        
    assert type(windows["-3"]) is list or windows["-3"] is None
    assert type(windows["-2"]) is dict or windows["-2"] is None
    assert type(windows["-1"]) is list or windows["-1"] is None
    assert type(windows["0"]) is dict
    assert type(windows["+1"]) is list or windows["+1"] is None
    assert type(windows["+2"]) is dict or windows["+2"] is None
    assert type(windows["+3"]) is list or windows["+3"] is None

    target_hunk_id = windows["0"]["id"]
    # convert them into labelled windows, both after edit verion and before edit version
    # labelled windows contains key: code_window, inline_labels, inter_labels, line_belong_to_hunk_id
    labelled_windows = {}
    for idx, window in windows.items():
        if window is None:
            labelled_windows[idx] = None
        if type(window) is list:
            labelled_windows[idx] = {
                "code_window": window,
                "inline_labels": ["keep"] * len(window),
                "inter_labels": ["null"] * (len(window) + 1),
                "to_insert": [],
                "line_belong_to_hunk_id": [-1] * len(window)
            }
            assert len(labelled_windows[idx]["code_window"]) == len(labelled_windows[idx]["inline_labels"])
            assert len(labelled_windows[idx]["code_window"]) + 1 == len(labelled_windows[idx]["inter_labels"])
            assert len(labelled_windows[idx]["code_window"]) == len(labelled_windows[idx]["line_belong_to_hunk_id"])
        elif type(window) is dict:
            if window["type"] == "insert":
                labelled_windows[idx] = {}
                labelled_windows[idx]["before"] = {
                    "code_window": window["before"],
                    "inline_labels": [],
                    "inter_labels": ["insert"],
                    "to_insert": [window["after"]],
                    "line_belong_to_hunk_id": [window["id"]] * len(window["before"])
                }
                labelled_windows[idx]["after"] = {
                    "code_window": window["after"],
                    "inline_labels": ["keep"] * len(window["after"]),
                    "inter_labels": ["null"] * (len(window["after"]) + 1),
                    "to_insert": [],
                    "line_belong_to_hunk_id": [-1] * len(window["after"])
                }
                assert len(labelled_windows[idx]["before"]["code_window"]) == len(labelled_windows[idx]["before"]["inline_labels"])
                assert len(labelled_windows[idx]["before"]["code_window"]) + 1 == len(labelled_windows[idx]["before"]["inter_labels"])
                assert len(labelled_windows[idx]["before"]["code_window"]) == len(labelled_windows[idx]["before"]["line_belong_to_hunk_id"])
                assert len(labelled_windows[idx]["after"]["code_window"]) == len(labelled_windows[idx]["after"]["inline_labels"])
                assert len(labelled_windows[idx]["after"]["code_window"]) + 1 == len(labelled_windows[idx]["after"]["inter_labels"])
                assert len(labelled_windows[idx]["after"]["code_window"]) == len(labelled_windows[idx]["after"]["line_belong_to_hunk_id"])
            elif window["type"] == "delete":
                labelled_windows[idx] = {}
                labelled_windows[idx]["before"] = {
                    "code_window": window["before"],
                    "inline_labels": ["delete"] * len(window["before"]),
                    "inter_labels": ["null"] * (len(window["before"]) + 1),
                    "to_insert": [],
                    "line_belong_to_hunk_id": [window["id"]] * len(window["before"])
                }
                labelled_windows[idx]["after"] = {
                    "code_window": window["after"],
                    "inline_labels": [],
                    "inter_labels": ["null"],
                    "to_insert": [],
                    "line_belong_to_hunk_id": [-1] * len(window["after"])
                }
                assert len(labelled_windows[idx]["before"]["code_window"]) == len(labelled_windows[idx]["before"]["inline_labels"])
                assert len(labelled_windows[idx]["before"]["code_window"]) + 1 == len(labelled_windows[idx]["before"]["inter_labels"])
                assert len(labelled_windows[idx]["before"]["code_window"]) == len(labelled_windows[idx]["before"]["line_belong_to_hunk_id"])
                assert len(labelled_windows[idx]["after"]["code_window"]) == len(labelled_windows[idx]["after"]["inline_labels"])
                assert len(labelled_windows[idx]["after"]["code_window"]) + 1 == len(labelled_windows[idx]["after"]["inter_labels"])
                assert len(labelled_windows[idx]["after"]["code_window"]) == len(labelled_windows[idx]["after"]["line_belong_to_hunk_id"])
            elif window["type"] == "replace":
                labelled_windows[idx] = {}
                labelled_windows[idx]["before"] = {
                    "code_window": [],
                    "inline_labels": [],
                    "inter_labels": [],
                    "to_insert": [],
                    "line_belong_to_hunk_id": [window["id"]] * len(window["before"])
                }
                labelled_windows[idx]["after"] = {
                    "code_window": [],
                    "inline_labels": [],
                    "inter_labels": [],
                    "to_insert": [],
                    "line_belong_to_hunk_id": [-1] * len(window["after"])
                }
                # get the after version first
                for block_idx, block in enumerate(window["blocks"]):
                    if type(block) is str:
                        labelled_windows[idx]["after"]["code_window"].append(block)
                        labelled_windows[idx]["after"]["inline_labels"].append("keep")
                        labelled_windows[idx]["after"]["inter_labels"].append("null")
                    else:
                        labelled_windows[idx]["after"]["code_window"].extend(block["after"])
                        labelled_windows[idx]["after"]["inline_labels"].extend(["keep"] * len(block["after"]))
                        labelled_windows[idx]["after"]["inter_labels"].extend(["null"] * len(block["after"]))
                labelled_windows[idx]["after"]["inter_labels"].append("null")
                # get the before version
                to_insert = "null"
                for block_idx, block in enumerate(window["blocks"]):
                    if type(block) is str:
                        labelled_windows[idx]["before"]["code_window"].append(block)
                        labelled_windows[idx]["before"]["inline_labels"].append("keep")
                        labelled_windows[idx]["before"]["inter_labels"].append(to_insert)
                        to_insert = "null"
                    else:
                        if block["block_type"] == "insert":
                            to_insert = "insert"
                            labelled_windows[idx]["before"]["to_insert"].append(block["after"])
                        elif block["block_type"] == "delete":
                            labelled_windows[idx]["before"]["code_window"].extend(block["before"])
                            labelled_windows[idx]["before"]["inline_labels"].extend(["delete"] * len(block["before"]))
                            labelled_windows[idx]["before"]["inter_labels"].extend([to_insert] + ["null"] * (len(block["before"]) - 1))
                            to_insert = "null"
                        elif block["block_type"] == "modify":
                            labelled_windows[idx]["before"]["code_window"].extend(block["before"])
                            labelled_windows[idx]["before"]["inline_labels"].extend(["replace"] * len(block["before"]))
                            if block_idx != 0 and window["blocks"][block_idx - 1]["block_type"] == "modify":
                                labelled_windows[idx]["before"]["inter_labels"].extend(["block-split"] + ["null"] * (len(block["before"]) - 1))
                            else:
                                labelled_windows[idx]["before"]["inter_labels"].extend([to_insert] + ["null"] * (len(block["before"]) - 1))
                            to_insert = "null"
                labelled_windows[idx]["before"]["inter_labels"].append(to_insert)
                assert len(labelled_windows[idx]["before"]["code_window"]) == len(labelled_windows[idx]["before"]["inline_labels"])
                try:
                    assert len(labelled_windows[idx]["before"]["code_window"]) + 1 == len(labelled_windows[idx]["before"]["inter_labels"])
                except:
                    print(f"code window:\n{window}")
                    print(f"labelled windows {idx}, before version code window:\n{labelled_windows[idx]['before']['code_window']}")
                    print(f"labelled windows {idx}, before version inline labels:\n{labelled_windows[idx]['before']['inter_labels']}")
                    raise NotImplementedError
                assert len(labelled_windows[idx]["before"]["code_window"]) == len(labelled_windows[idx]["before"]["line_belong_to_hunk_id"])
                assert len(labelled_windows[idx]["after"]["code_window"]) == len(labelled_windows[idx]["after"]["inline_labels"])
                assert len(labelled_windows[idx]["after"]["code_window"]) + 1 == len(labelled_windows[idx]["after"]["inter_labels"])
                assert len(labelled_windows[idx]["after"]["code_window"]) == len(labelled_windows[idx]["after"]["line_belong_to_hunk_id"])

    required_keys = ["-3","-2","-1","0","+1","+2","+3"]
    assert all(key in labelled_windows for key in required_keys)
    # based on the length of target window (after edit), deduct the number of lines we need from window[-1], window[-2], window[+1], window[+2]
    target_window_after_edit_line_num = len(labelled_windows["0"]["after"]["code_window"])
    before_target_lines = int(np.ceil((10 - (target_window_after_edit_line_num % 10))/2))
    after_target_lines = int(np.floor((10 - (target_window_after_edit_line_num % 10))/2))
    # In this case, we need to borrow code from both -2 window and -1 window
    if labelled_windows["-2"] is not None and labelled_windows["-1"] is not None and before_target_lines > len(labelled_windows["-1"]["code_window"]):
        prev_window = []
        # need to borrow lines from window[-2], combine -2 & -1, with 2 versions, before edit and after edit
        # before_version
        if before_target_lines > len(labelled_windows["-1"]["code_window"]) + len(labelled_windows["-2"]["before"]["code_window"]) \
        and labelled_windows["-3"] is not None: # borrow some from window -3
            lines_num_from_window_minus_3 = min(len(labelled_windows["-3"]["code_window"]), before_target_lines - len(labelled_windows["-1"]["code_window"]) - len(labelled_windows["-2"]["before"]["code_window"]))
            window_minus_3_code_window = labelled_windows["-3"]["code_window"][-lines_num_from_window_minus_3:]
            window_minus_3_inline_labels = labelled_windows["-3"]["inline_labels"][-lines_num_from_window_minus_3:]
            window_minus_3_inter_labels = labelled_windows["-3"]["inter_labels"][-lines_num_from_window_minus_3-1:]
            window_minus_2_code_window = labelled_windows["-2"]["before"]["code_window"]
            window_minus_2_inline_labels = labelled_windows["-2"]["before"]["inline_labels"]
            window_minus_2_inter_labels = labelled_windows["-2"]["before"]["inter_labels"]
            window_minus_2_to_insert = labelled_windows["-2"]["before"]["to_insert"]
            prev_window_bef_version = {
                "code_window": window_minus_3_code_window + window_minus_2_code_window + labelled_windows["-1"]["code_window"],
                "inline_labels": window_minus_3_inline_labels + window_minus_2_inline_labels + labelled_windows["-1"]["inline_labels"],
                "inter_labels": merge_inter_labels(merge_inter_labels(window_minus_3_inter_labels, window_minus_2_inter_labels), labelled_windows["-1"]["inter_labels"]),
                "to_insert": window_minus_2_to_insert,
                "line_belong_to_hunk_id": labelled_windows["-3"]["line_belong_to_hunk_id"][-lines_num_from_window_minus_3:] + labelled_windows["-2"]["before"]["line_belong_to_hunk_id"] + labelled_windows["-1"]["line_belong_to_hunk_id"]
            }
        else: # cannot borrow from -3 window, only borrow from -2 & -1
            lines_num_from_window_minus_2 = min(len(labelled_windows["-2"]["before"]["code_window"]), before_target_lines - len(labelled_windows["-1"]["code_window"]))
            window_minus_2_code_window = labelled_windows["-2"]["before"]["code_window"][-lines_num_from_window_minus_2:]
            window_minus_2_inline_labels = labelled_windows["-2"]["before"]["inline_labels"][-lines_num_from_window_minus_2:]
            window_minus_2_inter_labels = labelled_windows["-2"]["before"]["inter_labels"][-lines_num_from_window_minus_2-1:]
            insert_label_cnt = window_minus_2_inter_labels.count("insert")
            if insert_label_cnt != 0:
                window_minus_2_to_insert = labelled_windows["-2"]["before"]["to_insert"][-1*insert_label_cnt:]
            else:
                window_minus_2_to_insert = []
            prev_window_bef_version = {
                "code_window": window_minus_2_code_window + labelled_windows["-1"]["code_window"],
                "inline_labels": window_minus_2_inline_labels + labelled_windows["-1"]["inline_labels"],
                "inter_labels": merge_inter_labels(window_minus_2_inter_labels, labelled_windows["-1"]["inter_labels"]),
                "to_insert": window_minus_2_to_insert,
                "line_belong_to_hunk_id": labelled_windows["-2"]["before"]["line_belong_to_hunk_id"][-lines_num_from_window_minus_2:] + labelled_windows["-1"]["line_belong_to_hunk_id"]
            }
        assert len(prev_window_bef_version["code_window"]) == len(prev_window_bef_version["inline_labels"])
        assert len(prev_window_bef_version["code_window"]) + 1 == len(prev_window_bef_version["inter_labels"])
        assert len(prev_window_bef_version["code_window"]) == len(prev_window_bef_version["line_belong_to_hunk_id"])
        prev_window.append(prev_window_bef_version)

        # we remove after version, because this case will overlap with the case when
        # target hunk id = current target hunk id - 1
    elif labelled_windows["-1"] is not None: # only borrow from -1 window
        line_num = min(len(labelled_windows["-1"]["code_window"]), before_target_lines)
        cuted_prev_window = {
            "code_window": labelled_windows["-1"]["code_window"][-line_num:],
            "inline_labels": labelled_windows["-1"]["inline_labels"][-line_num:],
            "inter_labels": labelled_windows["-1"]["inter_labels"][-line_num-1:],
            "to_insert": [],
            "line_belong_to_hunk_id": [-1] * line_num
        }
        assert len(cuted_prev_window["code_window"]) == len(cuted_prev_window["inline_labels"])
        assert len(cuted_prev_window["code_window"]) + 1 == len(cuted_prev_window["inter_labels"])
        assert len(cuted_prev_window["code_window"]) == len(cuted_prev_window["line_belong_to_hunk_id"])
        prev_window = [cuted_prev_window]
    else:
        # both are None, borrow no lines
        prev_window = []

    # construct code window after target hunk
    if labelled_windows["+2"] is not None and labelled_windows["+1"] is not None and after_target_lines > len(labelled_windows["+1"]["code_window"]):
        # need to borrow lines from window[+2]
        next_window = []
        # before version
        if after_target_lines > len(labelled_windows["+1"]["code_window"]) + len(labelled_windows["+2"]["before"]["code_window"]) and labelled_windows["+3"] is not None:
            # borrow some from window +3
            lines_num_from_window_plus_3 = min(len(labelled_windows["+3"]["code_window"]), after_target_lines - len(labelled_windows["+1"]["code_window"]) - len(labelled_windows["+2"]["before"]["code_window"]))
            window_plus_3_code_window = labelled_windows["+3"]["code_window"][:lines_num_from_window_plus_3]
            window_plus_3_inline_labels = labelled_windows["+3"]["inline_labels"][:lines_num_from_window_plus_3]
            window_plus_3_inter_labels = labelled_windows["+3"]["inter_labels"][:lines_num_from_window_plus_3+1]
            window_plus_2_code_window = labelled_windows["+2"]["before"]["code_window"]
            window_plus_2_inline_labels = labelled_windows["+2"]["before"]["inline_labels"]
            window_plus_2_inter_labels = labelled_windows["+2"]["before"]["inter_labels"]
            window_plus_2_to_insert = labelled_windows["+2"]["before"]["to_insert"]
            next_window_bef_version = {
                "code_window": labelled_windows["+1"]["code_window"] + window_plus_2_code_window + window_plus_3_code_window,
                "inline_labels": labelled_windows["+1"]["inline_labels"] + window_plus_2_inline_labels + window_plus_3_inline_labels,
                "inter_labels": merge_inter_labels(merge_inter_labels(labelled_windows["+1"]["inter_labels"], window_plus_2_inter_labels), window_plus_3_inter_labels),
                "to_insert": window_plus_2_to_insert,
                "line_belong_to_hunk_id" : labelled_windows["+1"]["line_belong_to_hunk_id"] + labelled_windows["+2"]["before"]["line_belong_to_hunk_id"] + labelled_windows["+3"]["line_belong_to_hunk_id"][:lines_num_from_window_plus_3]
            }
        else: # cannot borrow from +3, only from +1 & +2
            lines_num_from_window_plus_2 = min(len(labelled_windows["+2"]["before"]["code_window"]), after_target_lines - len(labelled_windows["+1"]["code_window"]))
            window_plus_2_code_window = labelled_windows["+2"]["before"]["code_window"][:lines_num_from_window_plus_2]
            window_plus_2_inline_labels = labelled_windows["+2"]["before"]["inline_labels"][:lines_num_from_window_plus_2]
            window_plus_2_inter_labels = labelled_windows["+2"]["before"]["inter_labels"][:lines_num_from_window_plus_2+1]
            insert_label_cnt = window_plus_2_inter_labels.count("insert")
            if insert_label_cnt != 0:
                window_plus_2_to_insert = labelled_windows["+2"]["before"]["to_insert"][-1*insert_label_cnt:]
            else:
                window_plus_2_to_insert = []
            next_window_bef_version = {
                "code_window": labelled_windows["+1"]["code_window"] + window_plus_2_code_window,
                "inline_labels": labelled_windows["+1"]["inline_labels"] + window_plus_2_inline_labels,
                "inter_labels": merge_inter_labels(labelled_windows["+1"]["inter_labels"], window_plus_2_inter_labels),
                "to_insert": window_plus_2_to_insert,
                "line_belong_to_hunk_id" : labelled_windows["+1"]["line_belong_to_hunk_id"] + labelled_windows["+2"]["before"]["line_belong_to_hunk_id"][:lines_num_from_window_plus_2]
            }
        assert len(next_window_bef_version["code_window"]) == len(next_window_bef_version["inline_labels"])
        assert len(next_window_bef_version["code_window"]) + 1 == len(next_window_bef_version["inter_labels"])
        assert len(next_window_bef_version["code_window"]) == len(next_window_bef_version["line_belong_to_hunk_id"])
        next_window.append(next_window_bef_version)

        # after version
        if after_target_lines > len(labelled_windows["+1"]["code_window"]) + len(labelled_windows["+2"]["after"]["code_window"]) and labelled_windows["+3"] is not None:
            # borrow some from window +3
            lines_num_from_window_plus_3 = min(len(labelled_windows["+3"]["code_window"]), after_target_lines - len(labelled_windows["+1"]["code_window"]) - len(labelled_windows["+2"]["after"]["code_window"]))
            window_plus_3_code_window = labelled_windows["+3"]["code_window"][:lines_num_from_window_plus_3]
            window_plus_3_inline_labels = labelled_windows["+3"]["inline_labels"][:lines_num_from_window_plus_3]
            window_plus_3_inter_labels = labelled_windows["+3"]["inter_labels"][:lines_num_from_window_plus_3+1]
            window_plus_2_code_window = labelled_windows["+2"]["after"]["code_window"]
            window_plus_2_inline_labels = labelled_windows["+2"]["after"]["inline_labels"]
            window_plus_2_inter_labels = labelled_windows["+2"]["after"]["inter_labels"]
            next_window_aft_version = {
                "code_window": labelled_windows["+1"]["code_window"] + window_plus_2_code_window + window_plus_3_code_window,
                "inline_labels": labelled_windows["+1"]["inline_labels"] + window_plus_2_inline_labels + window_plus_3_inline_labels,
                "inter_labels": merge_inter_labels(merge_inter_labels(labelled_windows["+1"]["inter_labels"], window_plus_2_inter_labels), window_plus_3_inter_labels),
                "to_insert": [],
                "line_belong_to_hunk_id" : labelled_windows["+1"]["line_belong_to_hunk_id"] + labelled_windows["+2"]["after"]["line_belong_to_hunk_id"] + labelled_windows["+3"]["line_belong_to_hunk_id"][:lines_num_from_window_plus_3]
            }
        else:
            lines_num_from_window_plus_2 = min(len(labelled_windows["+2"]["after"]["code_window"]), after_target_lines - len(labelled_windows["+1"]["code_window"]))
            window_plus_2_code_window = labelled_windows["+2"]["after"]["code_window"][:lines_num_from_window_plus_2]
            window_plus_2_inline_labels = labelled_windows["+2"]["after"]["inline_labels"][:lines_num_from_window_plus_2]
            window_plus_2_inter_labels = labelled_windows["+2"]["after"]["inter_labels"][:lines_num_from_window_plus_2+1]
            next_window_aft_version = {
                "code_window": labelled_windows["+1"]["code_window"] + window_plus_2_code_window,
                "inline_labels": labelled_windows["+1"]["inline_labels"] + window_plus_2_inline_labels,
                "inter_labels": merge_inter_labels(labelled_windows["+1"]["inter_labels"], window_plus_2_inter_labels),
                "to_insert": [],
                "line_belong_to_hunk_id" : labelled_windows["+1"]["line_belong_to_hunk_id"] + labelled_windows["+2"]["after"]["line_belong_to_hunk_id"][:lines_num_from_window_plus_2]
            }
        assert len(next_window_aft_version["code_window"]) == len(next_window_aft_version["inline_labels"])
        assert len(next_window_aft_version["code_window"]) + 1 == len(next_window_aft_version["inter_labels"])
        assert len(next_window_aft_version["code_window"]) == len(next_window_aft_version["line_belong_to_hunk_id"])
        next_window.append(next_window_aft_version)
    elif labelled_windows["+1"] is not None:
        # only borrow lines from window[+1]
        line_num = min(len(labelled_windows["+1"]["code_window"]), after_target_lines)
        cuted_next_window = {
            "code_window": labelled_windows["+1"]["code_window"][:line_num],
            "inline_labels": labelled_windows["+1"]["inline_labels"][:line_num],
            "inter_labels": labelled_windows["+1"]["inter_labels"][:line_num+1],
            "to_insert": [],
            "line_belong_to_hunk_id": labelled_windows["+1"]["line_belong_to_hunk_id"][:line_num]
        }
        assert len(cuted_next_window["code_window"]) == len(cuted_next_window["inline_labels"])
        assert len(cuted_next_window["code_window"]) + 1 == len(cuted_next_window["inter_labels"])
        assert len(cuted_next_window["code_window"]) == len(cuted_next_window["line_belong_to_hunk_id"])
        next_window = [cuted_next_window]
    else:
        # both are None, borrow no lines
        next_window = []

    windows = [prev_window, [labelled_windows["0"]["after"]], next_window]
    cartesian_product_windows = list(itertools.product(*windows))
    
    type_3_windows = []
    for window_triplet in cartesian_product_windows:
        big_window = combine_windows(window_triplet)
        small_windows = split_window(big_window, 10, file_path, target_hunk_id)
        type_3_windows.extend(small_windows)
    return type_3_windows

def make_dataset(lang, dataset_name: str, snapshots_by_commit = None, auto_save = True):
    if snapshots_by_commit is None:
        with open(os.path.join(ROOT_PATH, "qualified_commit", f"{lang}_qualified_commit_snapshots.json"), "r") as f:
            snapshots_by_commit = json.load(f)
    
    with open(os.path.join(ROOT_PATH, "commit_info", f"{lang}_commit_info.jsonl"), "r") as f:
        commits_info = [json.loads(line) for line in f.readlines()]
    
    dataset = {}
    rejcted_commit_cnt = 0
    llama3, llama3_tokenizer = load_llama3()
    for commit_idx, (commit_url, snapshots) in enumerate(tqdm(snapshots_by_commit.items())):
        dataset[commit_url] = {}
        # find commit msg
        for commit_info in commits_info:
            if commit_url == commit_info["html_url"]:
                commit_msg = commit_info["commit"]["message"]
                break
        try:
            dataset[commit_url]["commit_msg"] = filter_clean_msg_with_llama(commit_msg, llama3, llama3_tokenizer)
            dataset[commit_url]["original_commit_msg"] = commit_msg
        except:
            dataset.pop(commit_url)
            rejcted_commit_cnt += 1
            continue
        
        # assign id to each hunk
        hunk_id = 0
        for file_path, snapshot in snapshots.items():
            for window in snapshot:
                if type(window) is dict:
                    window["id"] = hunk_id
                    hunk_id += 1
        
        dataset[commit_url]["hunks"] = []
        # make hunks (hunks are used for generator)
        for file_path, snapshot in snapshots.items():
            line_count = 0
            for window_idx, window in enumerate(snapshot):
                if type(window) is list:
                    line_count += len(window)
                elif type(window) is dict: 
                    hunk = {}
                    # find prior context and prior labels
                    if window_idx == 0: # if there's no prior context
                        prior_context = []
                        prior_inline_labels = []
                        prior_inter_labels = []
                    else:
                        prior_window = snapshot[window_idx - 1]
                        assert type(prior_window) is list
                        prior_context_lines = min(len(prior_window), random.choice([3, 4, 5]))
                        # extract the last few lines as prior context
                        prior_context = prior_window[-prior_context_lines:]
                        prior_inline_labels = ["keep"] * len(prior_context)
                        prior_inter_labels = ["null"] * len(prior_context)
                    # find posterior context and posterior labels
                    if window_idx == len(snapshot) - 1: # if there's no posterior context
                        posterior_context = []
                        posterior_inline_labels = []
                        posterior_inter_labels = []
                    else:
                        posterior_window = snapshot[window_idx + 1]
                        assert type(posterior_window) is list
                        posterior_context_lines = min(len(posterior_window), random.choice([3, 4, 5]))
                        # extract the first few lines as posterior context
                        posterior_context = posterior_window[:posterior_context_lines]
                        posterior_inline_labels = ["keep"] * len(posterior_context)
                        posterior_inter_labels = ["null"] * len(posterior_context)
                    hunk["id"] = window["id"]
                    target_window_len = len(window["before"])
                    if window["type"] == "insert":
                        target_code_window = []
                        target_inline_labels = []
                        target_inter_labels = ["insert"]
                    elif window["type"] == "delete":
                        target_code_window = window["before"]
                        target_inline_labels = ["delete"] * len(window["before"])
                        target_inter_labels = ["null"] * (len(window["before"]) + 1)
                    elif window["type"] == "replace":
                        target_code_window = window["blocks"]
                        target_inline_labels = []
                        target_inter_labels = []
                        insert_label = []
                        for block_idx, block in enumerate(window["blocks"]):
                            if block["block_type"] == "delete":
                                target_inline_labels += ["delete"] * len(block["before"])
                                target_inter_labels += insert_label + ["null"] * (len(block["before"]) - len(insert_label))
                                insert_label = []
                            elif block["block_type"] == "modify":
                                target_inline_labels += ["replace"] * len(block["before"])
                                if block_idx != 0 and window["blocks"][block_idx - 1]["block_type"] == "modify": 
                                    # if we have 2 consecutive modify blocks, use <block-split> label to separate them
                                    target_inter_labels += ["block-split"] + ["null"] * (len(block["before"]) - 1)
                                else:
                                    target_inter_labels += insert_label + ["null"] * (len(block["before"]) - len(insert_label))
                                insert_label = []
                            elif block["block_type"] == "insert":
                                insert_label = ["insert"]
                            else:
                                print(block["block_type"])
                                raise ValueError("Invalid block type")
                        target_inter_labels += insert_label + ["null"] * (1 - len(insert_label))
                    else:
                        raise ValueError("Invalid window type")
                    hunk["code_window"] = prior_context + target_code_window + posterior_context
                    hunk["inline_labels"] = prior_inline_labels + target_inline_labels + posterior_inline_labels
                    hunk["inter_labels"] = prior_inter_labels + target_inter_labels + posterior_inter_labels
                    code_window_len = len(prior_context) + target_window_len + len(posterior_context)
                    assert code_window_len == len(hunk["inline_labels"])
                    assert code_window_len + 1 == len(hunk["inter_labels"])
                    hunk["after_edit"] = window["after"]
                    hunk["file_path"] = file_path
                    hunk["type"] = window["type"]
                    hunk["edit_start_line_idx"] = line_count
                    line_count += len(window["before"])
                    dataset[commit_url]["hunks"].append(hunk)        

        # make sliding windows (sliding windows are used for edit locator)
        """
        Sliding window there's 3 types:
            1. Overlap with 1 or more edit hunk
            2. Overlap with 0 edit hunk, should be 1/3 of type 1
            3. What code looks like after edit has been applied, may mix with
               neighbour hunk that have not been edited
        """
        # sample type 1 sliding windows
        dataset[commit_url]["sliding_windows"] = []
        sliding_window_len = 10
        for file_path, snapshot in snapshots.items():
            line_count = 0
            sliding_window = { # initialize a sliding window
                "code_window": [],
                "inline_labels": [],
                "inter_labels": [],
                "overlap_hunk_ids": [],
                "to_insert": [],
                "file_path": file_path,
                "edit_start_line_idx": line_count
            }
            insert_label = "null"
            to_insert_content = None
            for window_idx, window in enumerate(snapshot):
                if type(window) is list:
                    for code_line in window:
                        # if existing sliding window is full, append it to dataset and create a new one
                        if len(sliding_window["code_window"]) == sliding_window_len:
                            sliding_window["inter_labels"].append(insert_label)
                            if insert_label == "insert":
                                assert to_insert_content is not None
                                sliding_window["to_insert"].append(to_insert_content)
                                # not update insert label because it's shared by 2 sliding windows inter labels
                                shared_insert_hunk_id = sliding_window["overlap_hunk_ids"][-1]
                            assert len(sliding_window["code_window"]) == len(sliding_window["inline_labels"])
                            assert len(sliding_window["code_window"]) + 1 == len(sliding_window["inter_labels"])
                            dataset[commit_url]["sliding_windows"].append(sliding_window)
                            sliding_window = { # initialize a sliding window
                                "code_window": [],
                                "inline_labels": [],
                                "inter_labels": [],
                                "overlap_hunk_ids": [],
                                "to_insert": [],
                                "file_path": file_path,
                                "edit_start_line_idx": line_count
                            }
                        sliding_window["code_window"].append(code_line)
                        sliding_window["inline_labels"].append("keep")
                        # here inter label indicate whether to insert code before each line
                        sliding_window["inter_labels"].append(insert_label)
                        if insert_label == "insert":
                            assert to_insert_content is not None
                            sliding_window["to_insert"].append(to_insert_content)
                            if shared_insert_hunk_id not in sliding_window["overlap_hunk_ids"]:
                                sliding_window["overlap_hunk_ids"].append(shared_insert_hunk_id)
                            insert_label = "null"
                            to_insert_content = None
                        line_count += 1
                elif type(window) is dict:
                    hunk_id = window["id"]
                    # case 1: it's an insert hunk
                    if window["type"] == "insert":
                        insert_label = "insert"
                        if hunk_id not in sliding_window["overlap_hunk_ids"]:
                            sliding_window["overlap_hunk_ids"].append(hunk_id)
                        shared_insert_hunk_id = hunk_id
                        assert to_insert_content is None
                        to_insert_content = window["after"]
                    # case 2: it's a delete hunk
                    elif window["type"] == "delete":
                        for code_line in window["before"]:
                            if len(sliding_window["code_window"]) == sliding_window_len:
                                sliding_window["inter_labels"].append(insert_label)
                                if insert_label == "insert":
                                    assert to_insert_content is not None
                                    sliding_window["to_insert"].append(to_insert_content)
                                    # not update insert label because it's shared by 2 sliding windows inter labels
                                    shared_insert_hunk_id = sliding_window["overlap_hunk_ids"][-1]
                                assert len(sliding_window["code_window"]) == len(sliding_window["inline_labels"])
                                assert len(sliding_window["code_window"]) + 1 == len(sliding_window["inter_labels"])
                                assert len(sliding_window["to_insert"]) == sliding_window["inter_labels"].count("insert")
                                dataset[commit_url]["sliding_windows"].append(sliding_window)
                                sliding_window = { # initialize a sliding window
                                    "code_window": [],
                                    "inline_labels": [],
                                    "inter_labels": [],
                                    "overlap_hunk_ids": [],
                                    "to_insert": [],
                                    "file_path": file_path,
                                    "edit_start_line_idx": line_count
                                }
                            if hunk_id not in sliding_window["overlap_hunk_ids"]:
                                sliding_window["overlap_hunk_ids"].append(hunk_id)
                            sliding_window["code_window"].append(code_line)
                            sliding_window["inline_labels"].append("delete")
                            sliding_window["inter_labels"].append(insert_label)
                            if insert_label == "insert":
                                assert to_insert_content is not None
                                sliding_window["to_insert"].append(to_insert_content)
                                if shared_insert_hunk_id not in sliding_window["overlap_hunk_ids"]:
                                    sliding_window["overlap_hunk_ids"].append(shared_insert_hunk_id)
                                insert_label = "null"
                                to_insert_content = None
                            line_count += 1
                    # case 3: it's a replace hunk
                    elif window["type"] == "replace":
                        for block_idx, block in enumerate(window["blocks"]):
                            if block["block_type"] == "delete":
                                for code_line in block["before"]:
                                    if len(sliding_window["code_window"]) == sliding_window_len:
                                        sliding_window["inter_labels"].append(insert_label)
                                        if insert_label == "insert":
                                            assert to_insert_content is not None
                                            sliding_window["to_insert"].append(to_insert_content)
                                            # not update insert label because it's shared by 2 sliding windows inter labels
                                            shared_insert_hunk_id = sliding_window["overlap_hunk_ids"][-1]
                                        assert len(sliding_window["code_window"]) == len(sliding_window["inline_labels"])
                                        assert len(sliding_window["code_window"]) + 1 == len(sliding_window["inter_labels"])
                                        assert len(sliding_window["to_insert"]) == sliding_window["inter_labels"].count("insert")
                                        dataset[commit_url]["sliding_windows"].append(sliding_window)
                                        sliding_window = { # initialize a sliding window
                                            "code_window": [],
                                            "inline_labels": [],
                                            "inter_labels": [],
                                            "overlap_hunk_ids": [],
                                            "to_insert": [],
                                            "file_path": file_path,
                                            "edit_start_line_idx": line_count
                                        }
                                    if hunk_id not in sliding_window["overlap_hunk_ids"]:
                                        sliding_window["overlap_hunk_ids"].append(hunk_id)
                                    sliding_window["code_window"].append(code_line)
                                    sliding_window["inline_labels"].append("delete")
                                    sliding_window["inter_labels"].append(insert_label)
                                    if insert_label == "insert":
                                        assert to_insert_content is not None
                                        sliding_window["to_insert"].append(to_insert_content)
                                        if shared_insert_hunk_id not in sliding_window["overlap_hunk_ids"]:
                                            sliding_window["overlap_hunk_ids"].append(shared_insert_hunk_id)
                                        insert_label = "null"
                                        to_insert_content = None
                                    line_count += 1
                            elif block["block_type"] == "insert":
                                insert_label = "insert"
                                if hunk_id not in sliding_window["overlap_hunk_ids"]:
                                    sliding_window["overlap_hunk_ids"].append(hunk_id)
                                shared_insert_hunk_id = hunk_id
                                assert to_insert_content is None
                                to_insert_content = block["after"]
                            elif block["block_type"] == "modify":
                                for code_line_idx, code_line in enumerate(block["before"]):
                                    if len(sliding_window["code_window"]) == sliding_window_len:
                                        sliding_window["inter_labels"].append(insert_label)
                                        if insert_label == "insert":
                                            assert to_insert_content is not None
                                            sliding_window["to_insert"].append(to_insert_content)
                                            # not update insert label because it's shared by 2 sliding windows inter labels
                                            shared_insert_hunk_id = sliding_window["overlap_hunk_ids"][-1]
                                        assert len(sliding_window["code_window"]) == len(sliding_window["inline_labels"])
                                        assert len(sliding_window["code_window"]) + 1 == len(sliding_window["inter_labels"])
                                        assert len(sliding_window["to_insert"]) == sliding_window["inter_labels"].count("insert")
                                        dataset[commit_url]["sliding_windows"].append(sliding_window)
                                        sliding_window = { # initialize a sliding window
                                            "code_window": [],
                                            "inline_labels": [],
                                            "inter_labels": [],
                                            "overlap_hunk_ids": [],
                                            "to_insert": [],
                                            "file_path": file_path,
                                            "edit_start_line_idx": line_count
                                        }
                                    if block_idx != 0 and window["blocks"][block_idx - 1]["block_type"] == "modify" and code_line_idx == 0 and len(sliding_window["code_window"]) != 0:
                                        # if we have 2 consecutive modify blocks, and we can see both of them in the same sliding window, use <block-split> label to separate them
                                        sliding_window["inter_labels"].append("block-split")
                                    else:
                                        sliding_window["inter_labels"].append(insert_label)
                                    if hunk_id not in sliding_window["overlap_hunk_ids"]:
                                        sliding_window["overlap_hunk_ids"].append(hunk_id)
                                    sliding_window["code_window"].append(code_line)
                                    sliding_window["inline_labels"].append("replace")
                                    if insert_label == "insert":
                                        assert to_insert_content is not None
                                        sliding_window["to_insert"].append(to_insert_content)
                                        if shared_insert_hunk_id not in sliding_window["overlap_hunk_ids"]:
                                            sliding_window["overlap_hunk_ids"].append(shared_insert_hunk_id)
                                        insert_label = "null"
                                        to_insert_content = None
                                    line_count += 1
        
        # Sample type 2 sliding windows and reduce their number of 1/3 of type 1
        all_sliding_windows = dataset[commit_url]["sliding_windows"]
        type2_sliding_windows = []
        type1_sliding_windows = []
        for sliding_window in all_sliding_windows:
            if sliding_window["overlap_hunk_ids"] == []:
                type2_sliding_windows.append(sliding_window)
            else:
                type1_sliding_windows.append(sliding_window)
        sample_number = max(1, len(type1_sliding_windows) // 3)
        sample_number = min(sample_number, len(type2_sliding_windows))
        type2_sliding_windows = random.sample(type2_sliding_windows, sample_number)
        # shuffle type 1 and type 2 sliding windows
        for sliding_window in type1_sliding_windows:
            sliding_window["sliding_window_type"] = "type1"
        for sliding_window in type2_sliding_windows:
            sliding_window["sliding_window_type"] = "type2"
        sampled_all_sliding_windows = type1_sliding_windows + type2_sliding_windows
        dataset[commit_url]["sliding_windows"] = sampled_all_sliding_windows
            
        # Make type 3 sliding windows
        type3_sliding_windows = []
        for hunk in dataset[commit_url]["hunks"]:
            edited_hunk_id = hunk["id"]
            # we believe that this hunk has been edited
            for file_path, snapshot in snapshots.items():
                for window_idx, window in enumerate(snapshot):
                    if type(window) is dict and window["id"] == edited_hunk_id: # we focus on the edited hunk
                        prior_3_window = snapshot[window_idx - 3] if window_idx > 2 else None
                        prior_2_window = snapshot[window_idx - 2] if window_idx > 1 else None
                        prior_1_window = snapshot[window_idx - 1] if window_idx > 0 else None
                        posterior_1_window = snapshot[window_idx + 1] if window_idx < len(snapshot) - 1 else None
                        posterior_2_window = snapshot[window_idx + 2] if window_idx < len(snapshot) - 2 else None
                        posterior_3_window = snapshot[window_idx + 3] if window_idx < len(snapshot) - 3 else None
                        neighbour_windows = {
                            "-3": prior_3_window,
                            "-2": prior_2_window,
                            "-1": prior_1_window,
                            "0": window,
                            "+1": posterior_1_window,
                            "+2": posterior_2_window,
                            "+3": posterior_3_window
                        }
                        type3_sliding_windows.extend(make_type3_sliding_window(neighbour_windows, file_path))
                    else:
                        continue
        # sample type3 sliding windows
        sample_number = max(1, len(type1_sliding_windows))
        sample_number = min(sample_number, len(type3_sliding_windows))
        type3_sliding_windows = random.sample(type3_sliding_windows, sample_number)
        dataset[commit_url]["sliding_windows"].extend(type3_sliding_windows)

    print(f"Rejected commit percentage: {rejcted_commit_cnt / len(snapshots_by_commit)}")
    if not os.path.exists(os.path.join(ROOT_PATH, dataset_name, lang)):
        os.makedirs(os.path.join(ROOT_PATH, dataset_name, lang)) 
    
    if auto_save:
        # extract 70% of dataset as training set, 10% as dev set, 20% as test sets
        train_dataset = {}
        dev_dataset = {}
        test_dataset = {}
        dataset_size = len(dataset)
        for idx, (commit_url, data) in enumerate(dataset.items()):
            if idx < dataset_size * 0.7:
                train_dataset[commit_url] = data
            elif idx < dataset_size * 0.8:
                dev_dataset[commit_url] = data
            else:
                test_dataset[commit_url] = data
        with open(os.path.join(ROOT_PATH, dataset_name, lang, "train.json"), "w") as f:
            json.dump(train_dataset, f, indent=4)
        with open(os.path.join(ROOT_PATH, dataset_name, lang, "dev.json"), "w") as f:
            json.dump(dev_dataset, f, indent=4)
        with open(os.path.join(ROOT_PATH, dataset_name, lang, "test.json"), "w") as f:
            json.dump(test_dataset, f, indent=4)        
    else:
        return dataset

if __name__ == "__main__":
    make_dataset("python", "dataset_fine_grain")