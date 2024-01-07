import os
import shutil
import pandas as pd
import numpy as np
from pytorch_grad_cam import GradCAM
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix


def shuffle():
    data = pd.read_excel("./shuffle_testdata_label.xlsx")
    data = data.rename(columns={"Unnamed: 0": "编号"})

    if os.path.exists("../shuffle_testdata") == False:
        os.makedirs("../shuffle_testdata")
    for i in range(len(data)):
        shutil.copy("../data" + data["img_path"].values[i][15:],
                    "../shuffle_testdata/" + str(data["编号"].values[i]) + data["img_path"].values[i][-4:])


def calculate_metrics(label, pred, average="weighted"):
    acc = (label == pred).sum() / len(label)
    recall = recall_score(label, pred, average=average)
    precision = precision_score(label, pred, average=average)
    f1 = f1_score(label, pred, average=average)

    return {"acc": acc, "recall": recall, "precision": precision, "f1_score": f1}


def refine(a):
    for i in range(len(a)):
        if a[i] != 0:
            a[i] -= 1

    return a


def binary_metrics(label, output1, pred):
    cm = confusion_matrix(label, pred)
    # print(cm)
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[0][0]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    acc = (label == pred).sum() / len(label)
    auc = roc_auc_score(label, output1)

    return {"acc": acc, "auc": auc, "recall": recall, "precision": precision, "specificity": specificity, "npv": npv, "f1_score": f1_score}



if __name__ == "__main__":
    f = open("../metrics.txt", "a+")
    human_result = pd.read_excel("../肛提肌断裂人工.xlsx")
    two_three_result = pd.read_excel("../two_three_result.xlsx")
    four_result = pd.read_excel("../four_result.xlsx")
    assert (human_result["lable_是否断裂_0非断裂；1断裂_"].values == two_three_result["two_label"].values).all()
    assert (refine(human_result["lable_断裂部位：左4；右3；双4_"].values) == two_three_result["four_label"].values).all()
    two_label = two_three_result["two_label"].values
    four_label = two_three_result["four_label"].values

    # human result(two)
    two_A_metrics = binary_metrics(two_label, human_result["A_二分类"].values, human_result["A_二分类"].values)
    two_B_metrics = binary_metrics(two_label, human_result["B_二分类"].values, human_result["B_二分类"].values)
    two_C_metrics = binary_metrics(two_label, human_result["C_二分类"].values, human_result["C_二分类"].values)

    print("two_A_metrics(二分类):", two_A_metrics)
    print("two_B_metrics(二分类):", two_B_metrics)
    print("two_C_metrics(二分类):", two_C_metrics)

    # human result(four)
    A_metrics = calculate_metrics(four_label, refine(human_result["A_四分类"].values))
    B_metrics = calculate_metrics(four_label, refine(human_result["B_四分类"].values))
    C_metrics = calculate_metrics(four_label, refine(human_result["C_四分类"].values))
    print("A_metrics(四分类):", A_metrics)
    print("B_metrics(四分类):", B_metrics)
    print("C_metrics(四分类):", C_metrics)

    # model result(two_three)
    two_metrics = binary_metrics(two_label, two_three_result["two_output1"].values,
                                 two_three_result["two_preds"].values)
    print("two_metrics(二分类):", two_metrics)

    two_three_metrics = calculate_metrics(four_label, two_three_result["four_preds"].values)
    print("two_three_metrics(四分类):", two_three_metrics)

    # model result(four)
    four_metrics = calculate_metrics(four_label, four_result["preds"].values)
    print("four_metrics(四分类):", four_metrics)

    all_metrics = pd.DataFrame([two_A_metrics, two_B_metrics, two_C_metrics,
                                A_metrics, B_metrics, C_metrics,
                                two_metrics, two_three_metrics, four_metrics])

    print(all_metrics)
    all_metrics.to_excel("../all_metrics.xlsx", index=False)
