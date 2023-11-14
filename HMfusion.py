import os
import shutil
import pandas as pd
import numpy as np

def shuffle():
    data = pd.read_excel("./shuffle_testdata_label.xlsx")
    data = data.rename(columns={"Unnamed: 0": "编号"})

    if os.path.exists("../shuffle_testdata") == False:
        os.makedirs("../shuffle_testdata")
    for i in range(len(data)):
        shutil.copy("../data"+data["img_path"].values[i][15:], "../shuffle_testdata/"+str(data["编号"].values[i])+data["img_path"].values[i][-4:])

if __name__ == "__main__":
    human_result = pd.read_excel("../肛提肌断裂人工.xlsx")
    A_four_acc = (human_result["lable_断裂部位：左4；右3；双4_"].values == human_result["A_四分类"].values).sum()/len(human_result)
    B_four_acc = (human_result["lable_断裂部位：左4；右3；双4_"].values == human_result["B_四分类"].values).sum()/len(human_result)
    C_four_acc = (human_result["lable_断裂部位：左4；右3；双4_"].values == human_result["C_四分类"].values).sum()/len(human_result)
    print("四分类acc：", [A_four_acc, B_four_acc, C_four_acc])