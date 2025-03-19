import os
import sys
import json

sys.path.append("../RQ5_simulation")
from is_clone import find_clone_in_project

def start_lsp(language, files_to_change, project_dir):
    # Initialize the LSP
    if language == "python":
        from LSPs.py_lsp import PyLanguageServer
        LSP = PyLanguageServer()
    elif language == "java":
        from LSPs.java_lsp import JavaLanguageServer
        LSP = JavaLanguageServer()
    elif language == "go":
        from LSPs.go_lsp import GoLanguageServer
        LSP = GoLanguageServer()
    elif language in ["javascript", "typescript"]:
        from LSPs.jsts_lsp import TsLanguageServer
        LSP = TsLanguageServer(language)
    else:
        raise ValueError(f"Language {language} not supported")
    
    LSP.initialize(project_dir)
    LSP.open_in_batch(files_to_change)
    
    return LSP

def ask_lsp(commit, LSP):
    """
    Make sure that commit.prev_edits contains only 1 edit, the target edit
    
    Return:
        identified_locations: a dict, key is the absolute file path, value is a list of identified locations
    """
    assert len(commit.prev_edits) == 1
    
    target_edit = commit.prev_edits[0]
    _, last_edit_start_at_line = commit.get_current_version(mode=target_edit["edit_type"], save=True)
    
    if target_edit["edit_type"] == "rename":
        identified_locations = {}
        deleted_identifiers = target_edit["edit_info"]["deleted_identifiers"]
        rename_map = target_edit["edit_info"]["map"]
        
        checked_renaming = []
        for identifier in deleted_identifiers:
            old_name = identifier["name"]
            new_name = rename_map[old_name]
            if f"{old_name} -> {new_name}" in checked_renaming:
                continue
            else:
                checked_renaming.append(f"{old_name} -> {new_name}")
            
            position = {
                "line": last_edit_start_at_line + identifier["start"][0],
                "character": identifier["start"][1]
            }
            
            last_edit_abs_file_path = os.path.normpath(os.path.join(commit.project_dir, commit.map[commit.prev_edits[-1]["idx"]]["at_file"]))
            response = LSP.rename(last_edit_abs_file_path, position, new_name)
            
            if len(response) == 0 or "error" in response[0] or response[0]["result"] is None :
                continue
            identified_locations = LSP._parse_rename_response(response, identified_locations, old_name, new_name)
        
        # with open("identified_locations.json", "w") as f:
        #     json.dump(identified_locations, f, indent=4)
        if last_edit_abs_file_path in identified_locations:
            for location in identified_locations[last_edit_abs_file_path]:
                if location["range"]["start"]["line"] == last_edit_start_at_line + deleted_identifiers[0]["start"][0]:
                    # remove the edit
                    identified_locations[last_edit_abs_file_path].remove(location)
                if location["range"]["start"]["line"] > last_edit_start_at_line + deleted_identifiers[0]["start"][0]:
                    # adjust the line number after the target edit has been edited
                    location["range"]["start"]["line"] += len(target_edit["after"]) - len(target_edit["before"])
                
        return identified_locations        
        
    elif target_edit["edit_type"] == "def&ref":
        last_edit_abs_file_path = os.path.normpath(os.path.join(commit.project_dir, commit.map[commit.prev_edits[-1]["idx"]]["at_file"]))
        position = {
            "line": last_edit_start_at_line + target_edit["edit_info"]["name_range_start"][0],
            "character": (target_edit["edit_info"]["name_range_start"][1] + target_edit["edit_info"]["name_range_end"][1]) // 2
        }
        response = LSP.references(last_edit_abs_file_path, position, wait_time=1)
        
        if len(response) == 0 or "error" in response[0] or response[0]["result"] is None:
            return None
        
        response = response[0]

        identified_locations = {}
        for location in response["result"]:
            if last_edit_abs_file_path == location["uri"][7:] and location["range"]["start"]["line"] == position["line"]:
                # this will be the last prior edit
                continue
            
            if location["uri"][7:] not in identified_locations:
                identified_locations[location["uri"][7:]] = []
            identified_locations[location["uri"][7:]].append(location)
            
        return identified_locations
    
    elif target_edit["edit_type"] == "clone":
        query = "".join(target_edit["before"])
        clone_locations = find_clone_in_project(commit, query, lsp_style=True)
        
        identified_locations = {}
        for location in clone_locations:
            abs_file_path = os.path.normpath(os.path.join(commit.project_dir, location["file_path"]))
            location.pop("file_path")
            
            if abs_file_path not in identified_locations:
                identified_locations[abs_file_path] = []
            identified_locations[abs_file_path].append(location)
        
        return identified_locations
        