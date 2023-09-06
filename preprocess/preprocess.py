import os
import shutil
import re



def rename(datapath):
    for root, dirs, files in os.walk(datapath, topdown=False):
        print(root,dirs)
        for dir in dirs:
            new_name = dir
            if "训练集" in dir:
                new_name = "TrainSet"
            elif "验证集" in dir:
                new_name = "TestSet"
            elif "左" in dir:
                new_name = "左侧"
            elif "右" in dir:
                new_name = "右侧"
            elif "双" in dir:
                new_name = "双侧"
            elif "断裂" in dir and "非" not in dir:
                new_name = "断裂"
            elif "非断" in dir:
                new_name = "非断裂"
            os.rename(os.path.join(root,dir), os.path.join(root, new_name))


def reorder(datapath, des_root, sup_datapath):
    duanlie_path = os.path.join(datapath, "断裂")
    for ce in ["左侧","右侧","双侧"]:
        ce_path = os.path.join(duanlie_path, ce)
        des_path = des_root+ce_path[len(datapath):]
        if not os.path.isdir(des_path):
            os.makedirs(des_path)
        file_list = os.listdir(ce_path)
        index = 1
        for i in range(len(file_list)):
            if file_list[i][-4:] in [".bmp",".jpg",".JPG",".Jpg",".png"]:
                shutil.copyfile(os.path.join(ce_path, file_list[i]), os.path.join(des_path, str(index)+file_list[i][-4:]))
                index += 1
        if sup_datapath != "":
            sup_ce_path = os.path.join(sup_datapath, ce)
            sup_file_list = os.listdir(sup_ce_path)
            for j in range(len(sup_file_list)):
                if sup_file_list[j][-4:] in [".bmp",".jpg",".JPG",".Jpg",".png"]:
                    shutil.copyfile(os.path.join(sup_ce_path, sup_file_list[j]), os.path.join(des_path, str(index)+sup_file_list[j][-4:]))
                    index += 1                    

    feiduanlie_path = os.path.join(datapath, "非断裂")
    des_path = des_root+feiduanlie_path[len(datapath):]
    if not os.path.isdir(des_path):
        os.makedirs(des_path)
    file_list = os.listdir(feiduanlie_path)
    index = 1
    for i in range(len(file_list)):
        if file_list[i][-4:] in [".bmp",".jpg",".JPG",".Jpg",".png"]:
            shutil.copyfile(os.path.join(feiduanlie_path, file_list[i]), os.path.join(des_path, str(index)+file_list[i][-4:]))
            index += 1


datapath = "../../original_data"
train_path = datapath+"/TrainSet"
test_path = datapath+"/TestSet"
des_train_path = "../../data/TrainSet"
des_test_path = "../../data/TestSet"
sup_datapath = "../../肛提肌断裂-补充训练集"

# rename(datapath)
if not os.path.isdir(des_train_path):
    os.makedirs(des_train_path)
if not os.path.isdir(des_test_path):
    os.makedirs(des_test_path)

reorder(train_path, des_train_path, sup_datapath)
for group in ["1.佛山市医", "2.湖南省妇幼", "3.广医三院", "4.白银", "5.陕西省人民医院"]:
    reorder(os.path.join(test_path, group), os.path.join(des_test_path, group), "")
