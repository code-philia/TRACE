import os
import re
import json
import time
import threading
import functools
import subprocess
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import List, Dict, Optional
from is_clone import find_clone_in_project
from utils import diagnostic_2_sliding_windows
from Locators import TRACE_make_locator_dataset, locator_predict

def timeout_decorator(timeout, timeout_return=None):
    """
    Decorator to add a timeout to any function.
    Returns `timeout_return` when the function exceeds the timeout.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result_container = {}
            exception_container = {}

            def target():
                try:
                    result_container['result'] = func(*args, **kwargs)
                except Exception as e:
                    exception_container['exception'] = e

            thread = threading.Thread(target=target)
            thread.daemon = True  # Set the thread as a daemon thread
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                print(f"Function '{func.__name__}' exceeded timeout of {timeout} seconds.")
                return timeout_return

            if 'exception' in exception_container:
                raise exception_container['exception']

            return result_container.get('result')
        return wrapper
    return decorator

class LanguageServer(ABC):
    def __init__(self, language_id: str, server_command: List[str], log: bool = False):
        """
        Initialize the language server process.
        """
        self.language_id = language_id
        self.process = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        self.request_id: int = 1
        self.log: bool = log
        self.messages: List[Dict] = []
        self.workspace_file_version: Dict[str, int] = {}

    def initialize(self, workspace_folders: list[str] | str, wait_time: float = 0.5):
        if isinstance(workspace_folders, str):
            workspace_folders = [workspace_folders]
        request_id = self._send_request(
            "initialize",
            params={
                "processId": None,
                "workspaceFolders": [
                    {
                        "uri": f"file://{workspace_folder}",
                        "name": f"Workspace {i}"
                    } for i, workspace_folder in enumerate(workspace_folders)
                ],
                "capabilities": self._get_capabilities()
            }
        )
        self._get_messages(request_id=request_id, message_num=1, wait_time=wait_time)
        self._send_notification("initialized")

    def _get_capabilities(self) -> Dict:
        """
        Get the capabilities for the language server.
        Should be overridden by subclasses if needed.
        """
        return {
            "textDocument": {
                "references": {"dynamicRegistration": True},
                "codeAction": {
                    "codeActionLiteralSupport": {
                        "codeActionKind": {
                            "valueSet": ["", "quickfix", "refactor", "refactor.extract", "refactor.inline",
                                       "refactor.rewrite", "source", "source.organizeImports"]
                        }
                    }
                }
            },
            "diagnostics": {
                "dynamicRegistration": True
            }
        }

    def did_open(self, file_path):
        with open(file_path, 'r') as f:
            file_content = f.read()

        self._send_notification(
            "textDocument/didOpen",
            params={
                "textDocument": {
                    "uri": f"file://{file_path}",
                    "languageId": self.language_id,
                    "version": 1,
                    "text": file_content
                }
            }
        )
        self.workspace_file_version[file_path] = 1
    
    def did_change(self, file_path: str):
        # 读取整个文件内容
        with open(file_path, 'r') as f:
            content = f.read()
        
        file_version = self.workspace_file_version.get(file_path, 0)
        self._send_notification(
            "textDocument/didChange",
            params={
                "textDocument": {
                    "uri": f"file://{file_path}",
                    "version": file_version + 1
                },
                "contentChanges": [
                    {
                        "text": content
                    }
                ]
            }
        )
        self.workspace_file_version[file_path] = file_version + 1
    
    def open_in_batch(self, file_paths: List[str]):
        for file_path in file_paths:
            try:
                self.did_open(file_path)
            except Exception as e:
                continue
            
    def rename(self, file_path: str, position: dict[str, int], new_name: str, wait_time: float = 0.5):
        if self.workspace_file_version.get(file_path, 0) == 0:
            self.did_open(file_path)
        else:
            self.did_change(file_path)
        
        request_id = self._send_request(
            "textDocument/rename",
            params={
                "textDocument": {
                    "uri": f"file://{file_path}"
                },
                "position": position,
                "newName": new_name
            }
        )
        messages = self._get_messages(request_id=request_id, message_num=1, wait_time=wait_time)
        return messages
    
    def references(self, file_path, position, wait_time: float = 0.5, include_declaration: bool = True):
        if self.workspace_file_version.get(file_path, 0) == 0:
            self.did_open(file_path)
        else:
            self.did_change(file_path)
        
        request_id = self._send_request(
            "textDocument/references",
            params={
                "textDocument": {
                    "uri": f"file://{file_path}"
                },
                "position": position,
                "context": {
                    "includeDeclaration": include_declaration
                }
            }
        )
        messages = self._get_messages(request_id=request_id, message_num=1, wait_time=wait_time)
        return messages
    
    def definitions(self, file_path, position, wait_time: float = 0.5):
        if self.workspace_file_version.get(file_path, 0) == 0:
            self.did_open(file_path)
        else:
            self.did_change(file_path)
            
        request_id = self._send_request(
            "textDocument/definition",
            params={
                "textDocument": {
                    "uri": f"file://{file_path}"
                },
                "position": position
            }
        )
        messages = self._get_messages(request_id=request_id, message_num=1, wait_time=wait_time)
        return messages
    
    def diagnostics(self, file_path, wait_time: float = 0.5):
        if self.workspace_file_version.get(file_path, 0) == 0:
            self.did_open(file_path)
        else:
            self.did_change(file_path)
        
        messages = self._get_messages(expect_method="textDocument/publishDiagnostics", message_num=1, wait_time=wait_time)
        return messages

    def close(self):
        request_id = self._send_request("shutdown")
        self._get_messages(request_id=request_id, message_num=1, wait_time=0.5)
        self._send_notification("exit")
        self.process.terminate()
        self.process.wait()
        print("Server closed")

    def get_all_file_paths(self, workspace_path: str) -> List[str]:
        file_paths = []
        for root, _, files in os.walk(workspace_path):
            for file in files:
                file_paths.append(os.path.join(root, file))
        return file_paths

    def _read_by_brace_matching(self, timeout: float = 0.1) -> Optional[str]:
        """
        Read a complete JSON message by matching the braces.
        
        Args:
            timeout: Timeout for select (seconds)
            
        Returns:
            Optional[str]: Return a complete JSON message string, or None if timeout
        """
        buffer = ""
        brace_count = 0
        inside_str = False
        escaped = False  # 处理转义字符
        target_start = '{"jsonrpc":"2.0"'
        
        while True:
            char = self.process.stdout.read(1)
            
            if buffer == "":
                assert char == "{"
                
            buffer += char
            
            if escaped:
                escaped = False
                continue
                
            if char == '\\':
                escaped = True
            elif char == '"':
                inside_str = not inside_str
            elif not inside_str:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
            if brace_count == 0:
                return buffer
        
    @timeout_decorator(timeout=5, timeout_return=None)
    def _read_lsp_messages(self, request_id: Optional[int] = None, expect_method: Optional[str] = None, message_num: Optional[int] = None, wait_time: Optional[float] = None):
        """
        Continuously read and parse JSON-RPC messages from the server's stdout.
        Messages are stored in the self.messages list.
        By specifying the `request_id` or `expect_method`, the function will stop when the message is received.
        If both parameters are set, the function will stop when either condition is met.
        """
        buffer = ""
        start_time = time.time()
        while True:
            line = self.process.stdout.readline()
            if not line:  # Exit if no more output is available
                break
            buffer += line
            match = re.search(r"Content-Length: (\d+)", buffer)
            if match:
                self.process.stdout.readline()  # Skip the blank line
                message = self._read_by_brace_matching()
                try:
                    json_message = json.loads(message.strip())
                    # if is desired message, return none with expected message saved in self.messages
                    if self._is_desired_message(json_message, request_id, expect_method):
                        return None
                except json.JSONDecodeError as e:
                    raise Exception(f"JSON Parse Error: {e}, Original Message: {message}")
                buffer = ""  # Reset buffer after processing a message
            
            if wait_time is not None and (time.time() - start_time) >= wait_time:
                return None
            if message_num is not None and len(self.messages) >= message_num:
                return None 
    
    def _is_desired_message(self, json_message: Dict, request_id: Optional[int] = None, expect_method: Optional[str] = None) -> bool:
        if request_id is not None: # if request_id is specified, only add the message if it has the same request_id
            if "id" in json_message and json_message["id"] == request_id: # if the response is a request response
                self.messages.append(json_message)
                if self.log:
                    print(f"[RECEIVED] {json.dumps(json_message, indent=2, ensure_ascii=False)}")
                return True
            else:
                return False
        elif expect_method is not None: # if expect_method is specified, only add the message if it has the same method
            if "method" in json_message and json_message["method"] == expect_method:
                self.messages.append(json_message)
                if self.log:
                    print(f"[RECEIVED] {json.dumps(json_message, indent=2, ensure_ascii=False)}")
                return False
            else:
                return False
        elif request_id is not None and expect_method is not None:
            if "id" in json_message and json_message["id"] == request_id and "method" in json_message and json_message["method"] == expect_method:
                self.messages.append(json_message)
                if self.log:
                    print(f"[RECEIVED] {json.dumps(json_message, indent=2, ensure_ascii=False)}")
                return True
            else:
                return False
        else: # if request_id is not specified, add all messages
            self.messages.append(json_message)
            return True
        
    def _get_messages(self, request_id: Optional[int] = None, expect_method: Optional[str] = None, message_num: Optional[int] = None, wait_time: Optional[float] = None) -> List[Dict]:
        """
        Retrieve messages from the server based on specified conditions:
        - request_id: Stop when a specific request ID is received.
        - expect_method: Stop when a specific method is received.
        - message_num: Stop when a specific number of messages are received.
        - wait_time: Stop after the specified amount of time (in seconds).
        If both parameters are set, the function will stop when either condition is met.

        Args:
            request_id (Optional[int]): Request ID of the message to retrieve.
            expect_method (Optional[str]): Method of the message to retrieve.
            message_num (Optional[int]): Number of messages to retrieve.
            wait_time (Optional[float]): Time in seconds allowed to wait for messages.

        Returns:
            List[Dict]: A list of received JSON-RPC messages.
        """
        self._read_lsp_messages(request_id=request_id, expect_method=expect_method, message_num=message_num, 
        wait_time=wait_time)  # Read all current messages
        messages, self.messages = self.messages, []  # Return and clear message list
        return messages
    
    def _send_to_process(self, message: str):
        if os.name == "nt": # Windows
            # TODO: Just a speculation, not verified.
            self.process.stdin.write(f"Content-Length: {len(message)}\n\n{message}")
        elif os.name == "posix": # Linux, macOS
            self.process.stdin.write(f"Content-Length: {len(message)}\r\n\r\n{message}")
        self.process.stdin.flush()
        
    def _create_message(self, method: str, params: dict = None, is_request: bool = True) -> str:
        message_data = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if is_request:
            message_data["id"] = self.request_id
            self.request_id += 1
        if params:
            message_data["params"] = params

        return message_data

    def _send_notification(self, method: str, params: dict = None):
        notification = self._create_message(method, params, is_request=False)
        notification = json.dumps(notification)
        self._send_to_process(notification)

    def _send_request(self, method: str, params: dict = None):
        request = self._create_message(method, params, is_request=True)
        request_id = request["id"]
        request_json = json.dumps(request)
        self._send_to_process(request_json)
        return request_id
    
    @abstractmethod
    def _parse_rename_response(self, response, edits):
        """
        Parse the response of rename request and update the edits
        Implemented in subclasses
        """
        pass
    
    @abstractmethod
    def _filter_diagnostics(self, diagnostics, last_edit_at_range, init_diagnose_msg):
        """
        Filter out the diagnostics that are not very helpful
        If not implemented, return the original diagnostics
        """
        pass
    
    def process_rename(self, commit, rename_info):
        # STEP 1: get the current version of files
        _, last_edit_start_at_line = commit.get_current_version(mode="rename", save=True)
        
        # STEP 2: get the file path of the last edit
        last_edit_abs_file_path = os.path.join(commit.project_dir, commit.map[commit.prev_edits[-1]["idx"]]["at_file"])
        
        # STEP 3: get the positions of the renames
        edits = {} # key is the absolute file path, value is the list of edits
        locations_in_last_edit = []
        
        # Blindly invoking will send empty rename info into here, return in time
        if "map" not in rename_info:
            return None
        
        for old_name, new_name in rename_info["map"].items():
            # find the position of the old name
            for delete_identifier in rename_info["deleted_identifiers"]:
                if old_name == delete_identifier["name"]:
                    position = {
                        "line": delete_identifier["start"][0] + last_edit_start_at_line,
                        "character": (delete_identifier["start"][1] + delete_identifier["end"][1]) // 2
                    }
                    locations_in_last_edit.append({
                        "range": {
                            "start": {
                                "line": delete_identifier["start"][0] + last_edit_start_at_line,
                                "character": delete_identifier["start"][1]
                            },
                            "end": {
                                "line": delete_identifier["end"][0] + last_edit_start_at_line,
                                "character": delete_identifier["end"][1]
                            }
                        },
                        "newText": new_name
                    })
            try:
                response = self.rename(last_edit_abs_file_path, position, new_name, wait_time=1)
            except:
                print("Error in getting rename")
                time.sleep(5)
                return None
            
            if len(response) == 0 or "error" in response[0] or "result" not in response[0] or response[0]["result"] is None: 
                continue
            edits = self._parse_rename_response(response, edits, old_name, new_name)
        
        # STEP 4: filter out the last prior edit
        for abs_file_path, edits_in_file in edits.items():
            if abs_file_path != last_edit_abs_file_path:
                # if the rename at file not the same as the last edit file, it will not be the last edit
                continue
            filtered_edits = []
            
            for edit in edits_in_file:
                # if the rename location is in the last edit, remove it
                need_filter = False
                for location in locations_in_last_edit:
                    if edit["range"] == location["range"] and edit["newText"] == location["newText"]:
                        need_filter = True
                if not need_filter:
                    filtered_edits.append(edit)
            
            edits[abs_file_path] = filtered_edits
        
        edits = {k: v for k, v in edits.items() if v != []}
        if edits == {}:
            return None
        
        # STEP 5: construct list of edit operation labels
        if len(commit.prev_edits[-1]["before"]) != len(commit.prev_edits[-1]["after"]):
            offset = len(commit.prev_edits[-1]["after"]) - len(commit.prev_edits[-1]["before"])
            use_offset_after = last_edit_start_at_line
        else:
            offset = 0
            use_offset_after = 0
            
        predictions = {}
        # update the current version of files, the last simulated edit is now after edit version
        commit.get_current_version(save=True)
        for abs_file_path, edits_in_file in edits.items():
            relative_file_path = os.path.relpath(abs_file_path, commit.project_dir)
            if relative_file_path not in commit.changed_files:
                continue
            
            with open(abs_file_path, "r") as f:
                file_content = f.readlines()
                
            predictions[relative_file_path] = {
                "inline_predictions": ["<keep>"] * len(file_content),
                "inline_confidences": [1.0] * len(file_content),
                "inter_predictions": ["<null>"] * (len(file_content) + 1),
                "inter_confidences": [1.0] * (len(file_content) + 1),
                "inline_service": ["normal"] * len(file_content),
                "inter_service": ["normal"] * (len(file_content) + 1)
            }
            
            for edit in edits_in_file:
                start_line = edit["range"]["start"]["line"]
                end_line = edit["range"]["end"]["line"]
                
                # If in last edit file & before edit
                if abs_file_path == last_edit_abs_file_path and start_line >= use_offset_after:
                    start_line += offset
                    end_line += offset
                
                for line_idx in range(start_line, end_line+1):
                    predictions[relative_file_path]["inline_predictions"][line_idx] = "<replace>"
                    predictions[relative_file_path]["inline_service"][line_idx] = "rename"

        return predictions
        
    def process_def_ref(self, defuse_info, commit, models, args):
        # STEP 1: get the current version of files
        _, last_edit_start_at_line = commit.get_current_version(save=True)
        
        # STEP 2: get the file path of the last edit
        last_edit_file_path = os.path.join(commit.project_dir, commit.map[commit.prev_edits[-1]["idx"]]["at_file"])
        
        # Blindly invoking will send empty rename info into here, return in time
        if "name_range_start" not in defuse_info:
            return None
        
        # STEP 3: get the positions of the def&ref
        position = {
            "line": defuse_info["name_range_start"][0] + last_edit_start_at_line,
            "character": (defuse_info["name_range_start"][1] + defuse_info["name_range_end"][1]) // 2
        }
        # print(f"last_edit_file_path: {last_edit_file_path}, position: {position}")
        try:
            response = self.references(last_edit_file_path, position, wait_time=1)
        except:
            print("Error in getting def&use")
            time.sleep(5)
            return None
        
        if response == []:
            return None
        else:
            response = response[0]

        if "error" in response or "result" not in response or response["result"] is None or response["result"] == []:
            return None
        
        # STEP 4: filter out the last prior edit
        identified_locations = []
        for location in response["result"]:
            if last_edit_file_path == location["uri"][7:] and location["range"]["start"]["line"] == position["line"]:
                # this will be the last prior edit
                continue
            location["file_path"] = os.path.relpath(location["uri"][7:], commit.project_dir)
            if location["file_path"] not in commit.changed_files:
                continue
            identified_locations.append(location)
        # print(f"identified locations:\n{json.dumps(identified_locations, indent=4)}")
        
        # STEP 5: construct sliding windows for identified locations
        sliding_windows_with_info = diagnostic_2_sliding_windows(identified_locations, commit)
        sliding_windows = [sliding_window["code_window"] for sliding_window in sliding_windows_with_info]
        # with open("debug_sw.json","w") as f:
        #     json.dump(sliding_windows,f,indent=4)
        if len(sliding_windows) == 0:
            return None
        
        # STEP 6: construct dataset for line locator
        dataset = TRACE_make_locator_dataset(sliding_windows, commit, models, args)
        dataloader = DataLoader(dataset, batch_size=args.locator_batch_size, shuffle=False)
        
        # STEP 7: feed into line locator to label the code window
        locator = models["locator"]
        locator_tokenizer = models["locator_tokenizer"]
        predicted_labels, predicted_confidences = locator_predict(locator, locator_tokenizer, dataloader, args, flatten=False)
        # print(f"predicted_labels: {predicted_labels}")

        # STEP 8: get the labels for each line of code in the files
        predictions = {}
        for relative_file_path in commit.changed_files:
            absolute_file_path = os.path.join(commit.project_dir, relative_file_path)
            with open(absolute_file_path, "r") as f:
                file_content = f.readlines()
            predictions[relative_file_path] = {
                "inline_predictions": ["<keep>"] * len(file_content),
                "inline_confidences": [1.0] * len(file_content),
                "inter_predictions": ["<null>"] * (len(file_content) + 1),
                "inter_confidences": [1.0] * (len(file_content) + 1),
                "inline_service": ["normal"] * len(file_content),
                "inter_service": ["normal"] * (len(file_content) + 1)
            }
        
        defref_find_location = False
        for window_info, predicted_label, predicted_confidence in zip(sliding_windows_with_info, predicted_labels, predicted_confidences):
            file_path = window_info["file_path"]
            start_line = window_info["start_line_idx"]
            end_line = start_line + len(window_info["code_window"])
            if file_path not in commit.changed_files:
                continue
            
            predicted_inter_labels = [label for i, label in enumerate(predicted_label) if i % 2 == 0]
            predicted_inline_labels = [label for i, label in enumerate(predicted_label) if i % 2 == 1]
            predicted_inter_confidences = [confidence for i, confidence in enumerate(predicted_confidence) if i % 2 == 0]
            predicted_inline_confidences = [confidence for i, confidence in enumerate(predicted_confidence) if i % 2 == 1]
            
            for label_idx, line_idx in enumerate(range(start_line, end_line)):
                predictions[file_path]["inline_predictions"][line_idx] = predicted_inline_labels[label_idx]
                predictions[file_path]["inline_confidences"][line_idx] = predicted_inline_confidences[label_idx]
                if predicted_inline_labels[label_idx] != "<keep>":
                    # print(f"{file_path}:{line_idx} have label: {predicted_inline_labels[label_idx]}")
                    predictions[file_path]["inline_service"][line_idx] = "def&ref"
                    defref_find_location = True
                
            for label_idx, line_idx in enumerate(range(start_line, end_line + 1)):
                predictions[file_path]["inter_predictions"][line_idx] = predicted_inter_labels[label_idx]
                predictions[file_path]["inter_confidences"][line_idx] = predicted_inter_confidences[label_idx]
                if predicted_inter_labels[label_idx] != "<null>":
                    predictions[file_path]["inter_service"][line_idx] = "def&ref"
                    defref_find_location = True
        
        if defref_find_location:
            return predictions
        else:
            return None
    
    def process_code_clone(self, query, commit, models, args):
        # in case blindly invoking will input unacceptable query
        if not isinstance(query, str):
            return None
        
        # STEP 1: update the current version of files
        _, last_edit_start_at_line = commit.get_current_version(save=True)
        
        # STEP 2: use code clone detector to find suspicious code
        clone_locations = find_clone_in_project(commit, query, lsp_style=True)
        sliding_windows_with_info = diagnostic_2_sliding_windows(clone_locations, commit)
        sliding_windows = [sliding_window["code_window"] for sliding_window in sliding_windows_with_info]
        
        if len(sliding_windows) == 0:
            return None
        
        # STEP 3: construct dataset for line locator
        dataset = TRACE_make_locator_dataset(sliding_windows, commit, models, args)
        dataloader = DataLoader(dataset, batch_size=args.locator_batch_size, shuffle=False)
    
        # STEP 4: feed into line locator to label the code window
        locator = models["locator"]
        locator_tokenizer = models["locator_tokenizer"]
        predicted_labels, predicted_confidences = locator_predict(locator, locator_tokenizer, dataloader, args, flatten=False)
        
        # STEP 5: get the labels for each line of code in the files
        predictions = {}
        for relative_file_path in commit.changed_files:
            absolute_file_path = os.path.join(commit.project_dir, relative_file_path)
            with open(absolute_file_path, "r") as f:
                file_content = f.readlines()
            predictions[relative_file_path] = {
                "inline_predictions": ["<keep>"] * len(file_content),
                "inline_confidences": [1.0] * len(file_content),
                "inter_predictions": ["<null>"] * (len(file_content) + 1),
                "inter_confidences": [1.0] * (len(file_content) + 1),
                "inline_service": ["normal"] * len(file_content),
                "inter_service": ["normal"] * (len(file_content) + 1)
            }
        
        find_clone_locations = False  
        for window_info, predicted_label, predicted_confidence in zip(sliding_windows_with_info, predicted_labels, predicted_confidences):
            file_path = window_info["file_path"]
            start_line = window_info["start_line_idx"]
            end_line = start_line + len(window_info["code_window"])
            
            predicted_inter_labels = [label for i, label in enumerate(predicted_label) if i % 2 == 0]
            predicted_inline_labels = [label for i, label in enumerate(predicted_label) if i % 2 == 1]
            predicted_inter_confidences = [confidence for i, confidence in enumerate(predicted_confidence) if i % 2 == 0]
            predicted_inline_confidences = [confidence for i, confidence in enumerate(predicted_confidence) if i % 2 == 1]
            
            for label_idx, line_idx in enumerate(range(start_line, end_line)):
                predictions[file_path]["inline_predictions"][line_idx] = predicted_inline_labels[label_idx]
                predictions[file_path]["inline_confidences"][line_idx] = predicted_inline_confidences[label_idx]
                if predicted_inline_labels[label_idx] != "<keep>":
                    predictions[file_path]["inline_service"][line_idx] = "clone"
                    find_clone_locations = True
                
            for label_idx, line_idx in enumerate(range(start_line, end_line + 1)):
                predictions[file_path]["inter_predictions"][line_idx] = predicted_inter_labels[label_idx]
                predictions[file_path]["inter_confidences"][line_idx] = predicted_inter_confidences[label_idx]
                if predicted_inter_labels[label_idx] != "<null>":
                    predictions[file_path]["inter_service"][line_idx] = "clone"
                    find_clone_locations = True
        
        if find_clone_locations:
            return predictions
        else:
            return None
        
    def process_diagnose(self, commit, models, args, return_diagnose=False):
        # STEP 1: get the current version of files
        _, last_edit_start_at_line = commit.get_current_version(save=True)
        last_edit_at_range = list(range(last_edit_start_at_line, last_edit_start_at_line + len(commit.prev_edits[-1]["after"])))
        
        # STEP 2: get the diagnostics of each file
        diagnostics = []
        for file_path in commit.changed_files:
            absolute_file_path = os.path.join(commit.project_dir, file_path)
            try:
                response = self.diagnostics(absolute_file_path, wait_time=1)
            except:
                print("Error in getting diagnostics")
                time.sleep(5)
                return None
            if response == []:
                continue
            else:
                response = response[0]
                
            if (response is None or \
                "params" not in response or \
                response["params"] is None or \
                "diagnostics" not in response["params"] or \
                response["params"]["diagnostics"] is None or \
                len(response["params"]["diagnostics"]) == 0
            ):
                continue
            if response["params"]["uri"][7:] != absolute_file_path:
                continue
            for diagnostic in response["params"]["diagnostics"]:
                diagnostic["file_path"] = file_path
                diagnostic["abs_file_path"] = absolute_file_path
            diagnostics.extend(response["params"]["diagnostics"])
        
        # STEP 3: filter out non-serious diagnostics
        diagnostics = self._filter_diagnostics(diagnostics, last_edit_at_range, args.init_diagnose_msg)
        if diagnostics is None or len(diagnostics) == 0:
            return None
        
        if return_diagnose:
            return diagnostics
        
        # STEP 4: construct sliding windows for diagnostics
        sliding_windows_with_info = diagnostic_2_sliding_windows(diagnostics, commit)
        sliding_windows = [sliding_window["code_window"] for sliding_window in sliding_windows_with_info]
        
        if len(sliding_windows) == 0:
            return None
        
        # STEP 5: construct dataset for line locator
        dataset = TRACE_make_locator_dataset(sliding_windows, commit, models, args)
        dataloader = DataLoader(dataset, batch_size=args.locator_batch_size, shuffle=False)
        
        # STEP 6: feed into line locator to label the code window
        locator = models["locator"]
        locator_tokenizer = models["locator_tokenizer"]
        predicted_labels, predicted_confidences = locator_predict(locator, locator_tokenizer, dataloader, args, flatten=False)
        
        # STEP 7: get the labels for each line of code in the files
        predictions = {}
        for relative_file_path in commit.changed_files:
            absolute_file_path = os.path.join(commit.project_dir, relative_file_path)
            with open(absolute_file_path, "r") as f:
                file_content = f.readlines()
            predictions[relative_file_path] = {
                "inline_predictions": ["<keep>"] * len(file_content),
                "inline_confidences": [1.0] * len(file_content),
                "inter_predictions": ["<null>"] * (len(file_content) + 1),
                "inter_confidences": [1.0] * (len(file_content) + 1),
                "inline_service": ["normal"] * len(file_content),
                "inter_service": ["normal"] * (len(file_content) + 1)
            }
        
        diagnose_find_location = False
        for window_info, predicted_label, predicted_confidence in zip(sliding_windows_with_info, predicted_labels, predicted_confidences):
            file_path = window_info["file_path"]
            start_line = window_info["start_line_idx"]
            end_line = start_line + len(window_info["code_window"])
            
            predicted_inter_labels = [label for i, label in enumerate(predicted_label) if i % 2 == 0]
            predicted_inline_labels = [label for i, label in enumerate(predicted_label) if i % 2 == 1]
            predicted_inter_confidences = [confidence for i, confidence in enumerate(predicted_confidence) if i % 2 == 0]
            predicted_inline_confidences = [confidence for i, confidence in enumerate(predicted_confidence) if i % 2 == 1]
            
            for label_idx, line_idx in enumerate(range(start_line, end_line)):
                predictions[file_path]["inline_predictions"][line_idx] = predicted_inline_labels[label_idx]
                predictions[file_path]["inline_confidences"][line_idx] = predicted_inline_confidences[label_idx]
                if predicted_inline_labels[label_idx] != "<keep>":
                    predictions[file_path]["inline_service"][line_idx] = "diagnose"
                    diagnose_find_location = True
                
            for label_idx, line_idx in enumerate(range(start_line, end_line + 1)):
                predictions[file_path]["inter_predictions"][line_idx] = predicted_inter_labels[label_idx]
                predictions[file_path]["inter_confidences"][line_idx] = predicted_inter_confidences[label_idx]
                if predicted_inter_labels[label_idx] != "<null>":
                    predictions[file_path]["inter_service"][line_idx] = "diagnose"
                    diagnose_find_location = True
        
        if diagnose_find_location:
            return predictions
        else:
            return None
        
    def extract_diagnose_msg(self, diagnoses):
        if diagnoses is None or len(diagnoses) == 0:
            return []
        msgs = []
        for diagnose in diagnoses:
            if diagnose["message"] not in msgs:
                msgs.append(diagnose["message"])
        return msgs
    