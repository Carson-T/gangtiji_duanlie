import os
import shutil
import re

datapath = "../../data/TrainSet/正中矢状切面-训练集"

# for i in os.listdir(datapath):
#     if i == "1.静息-标准":
#         new_name = "J_standard"
#     if i == "2.静息-非标准":
#         new_name = "J_nonstandard"
#     if i == "3.Valsalva-标准":
#         new_name = "V_standard"
#     if i == "4.Valsalva-非标准":
#         new_name = "V_nonstandard"
#     os.rename(os.path.join(datapath, i), os.path.join(datapath, new_name))

for dir in ["J_standard", "J_nonstandard", "V_standard", "V_nonstandard"]:
    class_path = os.path.join(datapath, dir)
    for file_name in os.listdir(class_path):
        idx = re.search("\d+", file_name).span()
        new_file_name = file_name[idx[0]:idx[1]] + ".bmp"
        os.rename(os.path.join(class_path, file_name), os.path.join(class_path, new_file_name))
