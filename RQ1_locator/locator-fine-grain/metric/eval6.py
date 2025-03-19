# One stop script to get all evaluation metrics for given prediction result
import json
import numpy as np

from tqdm import tqdm
from convert_label import label_conversion
from sklearn.metrics import classification_report, fbeta_score
    
def one_stop(model_path, dataset_path, lang):
    with open(f"{model_path}/{lang}/test_bm25.pred", "r") as f:
        preds = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
    with open(f"{model_path}/{lang}/test_bm25.gold", "r") as f:
        golds = [line.strip().split("\t")[1].split(" ") for line in f.readlines()]
    with open(f"{model_path}/{lang}/test_confidence_bm25.json", "r") as f:
        confidences = json.load(f)

    # Get the classification report if we convert back to 3 labels:
    converted_preds = []
    converted_golds = []
    converted_confs = []
    for pred, gold, confidence in zip(preds, golds, confidences):
        pred_inter_line = [pred[i] for i in range(0, len(pred), 2)]
        pred_inline = [pred[i] for i in range(1, len(pred), 2)]
        gold_inter_line = [gold[i] for i in range(0, len(gold), 2)]
        gold_inline = [gold[i] for i in range(1, len(gold), 2)]
        
        converted_label, converted_conf = label_conversion(pred_inline, pred_inter_line, confidence)
        converted_preds.extend(converted_label)
        converted_confs.extend(converted_conf)
        converted_golds.extend(label_conversion(gold_inline, gold_inter_line))
    print("==> Classification report with 3 labels:")
    print(classification_report(converted_golds, converted_preds, digits=4, labels=["<keep>", "<insert>", "<replace>"]), end="")
    fbeta = fbeta_score(converted_golds, converted_preds, beta=0.5, average=None, labels=["<keep>", "<replace>", "<insert>"])
    print(f"==> F0.5 score: {np.mean(fbeta):.4f}\n")
    
if __name__ == '__main__':
    model_path = "model_6"
    dataset_path = "./dataset_fine_grain"
    lang = "all"
    one_stop(model_path, dataset_path, lang)
