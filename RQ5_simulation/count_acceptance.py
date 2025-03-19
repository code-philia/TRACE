import json
import numpy as np

def main(file_path):
    with open(file_path, "r") as f:
        dataset = [json.loads(line) for line in f.readlines()]#[30:]
        
    cnt_1 = []
    cnt_3 = []
    cnt_5 = []
    for sample in dataset:
        for match, bleu in zip(sample["num_match_@1"], sample["bleu_@10"]):
            if match >= 1 and bleu == 100:
                cnt_1.append(1)
            else:
                cnt_1.append(0)
        
        for match, bleu in zip(sample["num_match_@3"], sample["bleu_@10"]):
            if match >= 1 and bleu == 100:
                cnt_3.append(1)
            else:
                cnt_3.append(0)
                
        for match, bleu in zip(sample["num_match_@5"], sample["bleu_@10"]):
            if match >= 1 and bleu == 100:
                cnt_5.append(1)
            else:
                cnt_5.append(0)
                
    print(f"Acceptance @1: {np.mean(cnt_1)*100:.2f}")
    print(f"Acceptance @3: {np.mean(cnt_3)*100:.2f}")
    print(f"Acceptance @5: {np.mean(cnt_5)*100:.2f}")
    

if __name__ == "__main__":
    file_name = "/home/user/workspace/trace/results/python/TRACE/2025-03-02_09-26-44.jsonl"
    main(file_name)
    print("=="*10)
    file_name = "/home/user/workspace/trace/results/python/woinvoker/2025-03-02_14-10-01.jsonl"
    main(file_name)
    print("=="*10)
    file_name = "/home/user/workspace/trace/results/python/enrich/2025-03-02_10-25-13.jsonl"
    main(file_name)
    print("=="*10)
    file_name = "/home/user/workspace/trace/results/python/plain/2025-03-02_11-28-24.jsonl"
    main(file_name)
    print("=="*10)
    file_name = "/home/user/workspace/trace/results/python/coedpilot/2025-03-03_13-01-07.jsonl"
    main(file_name)
    print("=="*10)
    file_name = "/home/user/workspace/trace/results/python/ccd/2025-03-02_17-28-10.jsonl"
    main(file_name)
    print("=="*10)