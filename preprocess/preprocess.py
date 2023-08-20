import os
import shutil
import re

datapath = "../../data/TestSet"
# sup_datapath = "../../肛提肌断裂-补充训练集"
preprocess_datapath = "../../preprocessed_data/TestSet"

groups = os.listdir(datapath)
for group in groups:
    duanlie_path = os.path.join(datapath, group)+"/断裂"
    for ce in ["左侧","右侧","双侧"]:
        ce_path = os.path.join(duanlie_path, ce)
        des_path = preprocess_datapath+ce_path[len(datapath):]
        if not os.path.isdir(des_path):
            os.makedirs(des_path)
        file_list = os.listdir(ce_path)
        index = 1
        for i in range(len(file_list)):
            if file_list[i][-4:] in [".bmp",".jpg",".JPG",".Jpg",".png"]:
                shutil.copyfile(os.path.join(ce_path, file_list[i]), os.path.join(des_path, str(index)+file_list[i][-4:]))
                index += 1

    feiduanlie_path = os.path.join(datapath, group)+"/非断裂"
    des_path = preprocess_datapath+feiduanlie_path[len(datapath):]
    if not os.path.isdir(des_path):
        os.makedirs(des_path)
    file_list = os.listdir(feiduanlie_path)
    index = 1
    for i in range(len(file_list)):
        if file_list[i][-4:] in [".bmp",".jpg",".JPG",".Jpg",".png"]:
            shutil.copyfile(os.path.join(feiduanlie_path, file_list[i]), os.path.join(des_path, str(index)+file_list[i][-4:]))
            index += 1
    # for j in range(len(sup_file_list)):
    #     shutil.copyfile(os.path.join(sup_group_path, sup_file_list[j]), os.path.join(des_group_path, str(j+1+len(file_list))+sup_file_list[j][-4:]))