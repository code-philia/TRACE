import json

from tqdm import tqdm
from code_window import CodeWindow
from transformers import RobertaTokenizer

ROOT_PATH = "/media/user"

def filter_by_length(lang, dataset_path, tokenizer_name, max_length=512):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    new_special_tokens = ["<inter-mask>",
                          "<code_window>", "</code_window>", 
                          "<prompt>", "</prompt>", 
                          "<prior_edits>", "</prior_edits>",
                          "<edit>", "</edit>",
                          "<keep>", "<replace>", "<delete>",
                          "<null>", "<insert>", "<block-split>",
                          "</insert>","<replace-by>", "</replace-by>"]
    tokenizer.add_tokens(new_special_tokens, special_tokens=True)
    for name in ["train", "dev", "test"]:
        with open(f"{ROOT_PATH}/{dataset_path}/{lang}/{name}.json", "r") as f:
            dataset = json.load(f)
        
        commit_contains_long_code_window = []    
        for commit_url, commit in tqdm(dataset.items()):
            for hunk in commit["hunks"]:
                hunk = CodeWindow(hunk, "hunk")
                input_seq = hunk.formalize_as_generator_target_window(False)
                if len(tokenizer.encode(input_seq)) > max_length - 10 and \
                commit_url not in commit_contains_long_code_window: # 10 is reserved margin
                    commit_contains_long_code_window.append(commit_url)
                    break # if one hunk is too long, then the whole commit is too long
            for sliding_window in commit["sliding_windows"]:
                sliding_window = CodeWindow(sliding_window, "sliding_window")
                input_seq = sliding_window.formalize_as_locator_target_window(False)[0]
                if len(tokenizer.encode(input_seq)) > max_length - 10 and \
                commit_url not in commit_contains_long_code_window: # 10 is reserved margin
                    commit_contains_long_code_window.append(commit_url)
                    break
        
        for commit_url in commit_contains_long_code_window:
            del dataset[commit_url]
        
        with open(f"{ROOT_PATH}/{dataset_path}/{lang}/{name}.json", "w") as f:
            json.dump(dataset, f, indent=4)
            
if __name__ == '__main__':
    lang = "go"
    filter_by_length(lang, "dataset_fine_grain", "salesforce/codet5-base")
    filter_by_length(lang, "dataset_fine_grain", "microsoft/codebert-base")
    