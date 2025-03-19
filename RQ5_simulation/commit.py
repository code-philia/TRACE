import os
import json
from utils import extract_hunks
from code_window import CodeWindow
from enriched_semantic import construct_prior_edit_hunk, finer_grain_window

class Commit:
    def __init__(self, commit_url):
        self.commit_url = commit_url
        self.project_name = commit_url.split("/")[-3]
        self.commit_message, self.snapshots = extract_hunks(commit_url)
        self.changed_files = []
        self.map = {} # map edit idx to file path and the window idx inside the file
        for file_path, snapshot in self.snapshots.items():
            self.changed_files.append(file_path)
            for window_idx, window in enumerate(snapshot):
                if isinstance(window, list):
                    continue
                self.map[window["idx"]] = {
                    "at_file": file_path,
                    "at_window": window_idx
                }
        self.prev_edits = []
        self.enriched_prev_edits = []
        self.language = None
        self.project_dir = os.path.join("/media/user/repos", self.project_name)
        
    def get_edit(self, idx):
        return self.snapshots[self.map[idx]["at_file"]][self.map[idx]["at_window"]]
    
    def get_edit_with_context(self, idx):
        edit_at_file, edit_idx_in_file = self.map[idx]["at_file"], self.map[idx]["at_window"]
        if edit_idx_in_file > 0:
            prefix_context = self.snapshots[edit_at_file][edit_idx_in_file - 1]
        else:
            prefix_context = []
        try:
            suffix_context = self.snapshots[edit_at_file][edit_idx_in_file + 1]
        except:
            suffix_context = []
        
        prefix_length = min(10, len(prefix_context))
        suffix_length = min(10, len(suffix_context))
        prefix_content = prefix_context[-prefix_length:]
        suffix_content = suffix_context[:suffix_length]
        
        edit = self.snapshots[edit_at_file][edit_idx_in_file].copy()
        edit["before"] = prefix_content + edit["before"] + suffix_content
        edit["after"] = prefix_content + edit["after"] + suffix_content
        return edit
        
    def hunk_num(self):
        return len(self.map)
    
    def get_current_version(self, mode=None, save=False):
        version = {}
        start_at_line = None
        for file_path, snapshot in self.snapshots.items():
            current_file = []
            line_cnt = 0
            for window in snapshot:
                if isinstance(window, list):
                    current_file.extend(window)
                    line_cnt += len(window)
                elif not window["simulated"]:
                    current_file.extend(window["before"])
                    line_cnt += len(window["before"])
                elif window["simulated"]: # if this edit has been simulated
                    if mode == "rename" and window["idx"] == self.prev_edits[-1]["idx"]:
                        # if mode is rename, we do not update, as we still need this location to find other occurences of this variable
                        current_file.extend(window["before"])
                        start_at_line = line_cnt
                    else: 
                        if window["idx"] == self.prev_edits[-1]["idx"]:
                            start_at_line = line_cnt
                        current_file.extend(window["after"])
                        line_cnt += len(window["after"])
                        
            version[file_path] = current_file
        
        if save:
            for file_path, file_content in version.items():
                with open(os.path.join(self.project_dir, file_path), "w") as f:
                    f.write("".join(file_content))
        
        return version, start_at_line

    def add_prev_edit(self, edit):
        # First add edit to prev_edits. This list represent edits in the most simplest way.
        self.prev_edits.append(edit)
        # Then add edit to enriched_prev_edits. This list represent edits in an enriched way.
        # self.enriched_prev_edits is used for TRACE, TRACE-wo-Invoker, EnrichedSemantics, PlainSemantics (reduced to 3 labels)
        edit_at_file = self.map[edit["idx"]]["at_file"]
        snapshot = self.snapshots[edit_at_file]
        edit_hunk = construct_prior_edit_hunk(snapshot, edit, self.language)
        edit_hunk = CodeWindow(edit_hunk, "hunk")
        self.enriched_prev_edits.append(edit_hunk)
        
        edits_in_file_all_simulated = True
        for window in self.snapshots[edit_at_file]:
            if isinstance(window, list):
                continue
            if not window["simulated"]:
                edits_in_file_all_simulated = False
                break
            
        if edits_in_file_all_simulated:
            self.changed_files.remove(edit_at_file)
        
    def get_gold_labels(self, args):
        """
        Given the current edit status, return the gold labels for each file and their each line of code for current snapshot
        
        Args:
            label_num: int, the number of labels for each line of code
        """
        gold_labels = {}
        if args.label_num == 3 and args.system != "CoEdPilot":
            # For PlainSemantics and CodeCloneDetector
            for file_path, snapshot in self.snapshots.items():
                inline_golds = []
                for window in snapshot:
                    if isinstance(window, list):
                        inline_golds.extend(["<keep>"] * len(window))
                    else:
                        if window["simulated"]:
                            inline_golds.extend(["<keep>"] * len(window["after"]))
                        else:
                            assert window["type"] in ["insert", "delete", "replace"]
                            if window["type"] == "insert":
                                inline_golds.pop()
                                inline_golds.append("<insert>")
                            else:
                                inline_golds.extend(["<replace>"] * len(window["before"]))
                gold_labels[file_path] = {
                    "inline_golds": inline_golds
                }
                
        elif args.label_num == 6:
            # For TRACE, TRACE-wo-Invoker, EnrichedSemantics
            for file_path, snapshot in self.snapshots.items():
                inline_golds = []
                inter_golds = []
                inter_label = "<null>"
                for window in snapshot:
                    if type(window) == list:
                        inline_golds.extend(["<keep>"] * len(window))
                        if inter_label == "<insert>":
                            inter_golds.append(inter_label)
                        else:
                            # if inter_label is <block-split>, we replace it with <null>
                            # as <block-split> is only valid between 2 <replace> blocks
                            inter_golds.append("<null>")
                        inter_label = "<null>"
                        inter_golds.extend(["<null>"] * (len(window)-1))
                    else:
                        if window["simulated"]: # this hunk has been edited
                            if window["type"] == "delete":
                                continue
                            inline_golds.extend(["<keep>"] * len(window["after"]))
                            assert inter_label == "<null>" # any hunk's previous window should be non-edit window (list), hence should have addressed the inter_label to <null>
                            inter_golds.append(inter_label)
                            inter_golds.extend(["<null>"] * (len(window["after"])-1))
                        else:
                            if window["type"] == "insert":
                                inter_label = "<insert>"
                            elif window["type"] == "delete":
                                inline_golds.extend(["<delete>"] * len(window["before"]))
                                assert inter_label == "<null>" # any hunk's previous window should be non-edit window (list), hence should have addressed the inter_label to <null>
                                inter_golds.append(inter_label)
                                inter_golds.extend(["<null>"] * (len(window["before"])-1))
                            else: # window of modify type
                                code_blocks = finer_grain_window(window["before"], window["after"], self.language)
                                for block in code_blocks:
                                    if block["block_type"] == "insert":
                                        inter_label = "<insert>"
                                    elif block["block_type"] == "delete":
                                        if inter_label == "<block-split>":
                                            inter_golds.append("<null>")
                                        else: 
                                            inter_golds.append(inter_label)
                                        inter_label = "<null>"
                                        inline_golds += ["<delete>"] * len(block["before"])
                                        inter_golds += ["<null>"] * (len(block["before"]) - 1)
                                    elif block["block_type"] == "modify":
                                        inter_golds.append(inter_label)
                                        inter_label = "<block-split>"
                                        inline_golds += ["<replace>"] * len(block["before"])
                                        inter_golds += ["<null>"] * (len(block["before"]) - 1)
                        
                if inter_label == "<block-split>":
                    inter_golds.append("<null>")
                else:
                    inter_golds.append(inter_label)
                    
                for label in inline_golds + inter_golds:
                    assert label[0] == "<" and label[-1] == ">"
                
                assert len(inline_golds) + 1 == len(inter_golds)
                
                gold_labels[file_path] = {
                    "inline_golds": inline_golds,
                    "inter_golds": inter_golds
                }
        
        elif args.system == "CoEdPilot":
            for file_path, snapshot in self.snapshots.items():
                inline_golds = []
                for window in snapshot:
                    if isinstance(window, list):
                        inline_golds.extend(["<keep>"] * len(window))
                        continue
                    if window["simulated"]:
                        inline_golds.extend(["<keep>"] * len(window["after"]))
                        continue
                    if window["type"] == "insert":
                        inline_golds.pop()
                        inline_golds.append("<add>")
                    else:
                        inline_golds.extend(["<replace>"] * len(window["before"]))
                gold_labels[file_path] = {
                    "inline_golds": inline_golds
                }
        
        else:
            raise ValueError(f"System {args.system} not supported")
        
        
        return gold_labels

    def unsimulated_edit_locations(self, args):
        """
        Return the locations of unsimulated edits
        """
        unsimulated_locations = []
        for file_path, snapshot in self.snapshots.items():
            line_idx = 0
            for window in snapshot:
                if isinstance(window, list):
                    line_idx += len(window)
                    continue
                if window["simulated"]:
                    line_idx += len(window["after"])
                else:
                    if window["before"] == []:
                        if args.label_num == 6:
                            unsimulated_locations.append({
                                "file_path": file_path,
                                "hunk_idx": window["idx"],
                                "line_idxs": [line_idx]
                            })
                        elif args.label_num == 3:
                            unsimulated_locations.append({
                                "file_path": file_path,
                                "hunk_idx": window["idx"],
                                "line_idxs": [line_idx-1]
                            })
                    else:
                        unsimulated_locations.append({
                            "file_path": file_path,
                            "hunk_idx": window["idx"],
                            "line_idxs": [i for i in range(line_idx, line_idx + len(window["before"]))]
                        })
                    line_idx += len(window["before"])
        return unsimulated_locations
    
    def unsimulated_edit_idxs(self):
        unsimulated_idxs = []
        for edit_idx in range(self.hunk_num()):
            edit = self.get_edit(edit_idx)
            if not edit["simulated"]:
                unsimulated_idxs.append(edit["idx"])
        return unsimulated_idxs
