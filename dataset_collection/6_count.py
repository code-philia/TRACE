import os
import json
from dotenv import load_dotenv

load_dotenv(".env")
ROOT_PATH = os.getenv("ROOT_PATH")

def count(lang: str, dataset_name: str):
    global ROOT_PATH
    
    with open(os.path.join(ROOT_PATH, dataset_name, lang, "train.json"), "r") as f:
        train = json.load(f)
    with open(os.path.join(ROOT_PATH, dataset_name, lang, "test.json"), "r") as f:
        test = json.load(f)
    with open(os.path.join(ROOT_PATH, dataset_name, lang, "dev.json"), "r") as f:
        dev = json.load(f)
    
    combined = {**train, **dev, **test}
    to_count = {
        "train": train,
        "dev": dev,
        "test": test,
        "combined": combined
    }
    for name, dataset in to_count.items():
        print(f"{lang} dataset: {name}")
        # count project number
        projects = set()
        for commit_url in dataset.keys():
            proj_name = commit_url.split('/')[-3]
            if proj_name not in projects:
                projects.add(proj_name)
        print(f"#Projects: {len(projects)}")
        
        # count commit number
        print(f"#Commits: {len(dataset)}")
        
        # count hunk number (generator)
        hunk_num = 0
        for commit_url, data in dataset.items():
            hunk_num += len(data['hunks'])
        print(f"#Hunks: {hunk_num}")
        
        # count sliding window number (locator)
        sld_win_num = 0
        inline_num = {
            "keep": 0,
            "delete": 0,
            "replace": 0
        }
        interline_num = {
            "null": 0,
            "insert": 0,
            "block-split": 0
        }
        for commit_url, data in dataset.items():
            sld_win_num += len(data['sliding_windows'])
            for sld_win in data['sliding_windows']:
                inline_labels = sld_win['inline_labels']
                inter_labels = sld_win['inter_labels']
                for label in inline_labels:
                    inline_num[label] += 1
                for label in inter_labels:
                    interline_num[label] += 1
        print(f"#Sliding Windows: {sld_win_num}")
        print(f"#Inline Labels: {inline_num}")
        print(f"#Interline Labels: {interline_num}\n")

if __name__ == "__main__":
    for lang in ["python", "go", "java", "javascript", "typescript", "all"]:
        count(lang, "dataset_fine_grain")
    
    