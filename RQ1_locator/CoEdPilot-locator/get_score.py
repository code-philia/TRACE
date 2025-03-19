import os
from sklearn.metrics import classification_report
def get_score(model_path, lang):
    with open(os.path.join(model_path, lang, "test_0.gold"), "r") as f:
        gold = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
        
    with open(os.path.join(model_path, lang, "test_0.output"), "r") as f:
        pred = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
    

    all_gold = []
    all_pred = []
    for g, p in zip(gold, pred):
        all_gold.extend(g)
        all_pred.extend(p)
        
    print(classification_report(all_gold, all_pred))
    
if __name__ == "__main__":
    model_path = "model"
    lang = "python"
    
    get_score(model_path, lang)