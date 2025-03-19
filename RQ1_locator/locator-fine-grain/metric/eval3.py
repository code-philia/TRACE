# One stop script to get all evaluation metrics for given prediction result
import json
import numpy as np

from sklearn.metrics import classification_report, fbeta_score

def one_stop(model_path, dataset_path, lang):
    with open(f"{model_path}/{lang}/test_bm25.pred", "r") as f:
        preds = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
    with open(f"{model_path}/{lang}/test_bm25.gold", "r") as f:
        golds = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
    with open(f"{model_path}/{lang}/test_confidence_bm25.json", "r") as f:
        confidences = json.load(f)
        
    # 1. Get Original Acc, F1, Precision, Recall
    flat_preds = [item for sublist in preds for item in sublist]
    flat_golds = [item for sublist in golds for item in sublist]
    print("==> Original classification report:")
    print(classification_report(flat_golds, flat_preds, digits=4, labels=["<keep>", "<insert>", "<replace>"]), end="")
    fbeta = fbeta_score(flat_golds, flat_preds, beta=0.5, average=None, labels=["<keep>", "<insert>", "<replace>"])
    print(f"==> F0.5 score: {np.mean(fbeta):.4f}\n")
    
if __name__ == '__main__':
    model_path = "model_3"
    dataset_path = "./dataset_fine_grain"
    lang = "all"
    one_stop(model_path, dataset_path, lang)
