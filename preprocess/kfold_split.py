import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2023)

img_paths = []
labels = []

# # duanlie 
# root_path = "../../data_3subimg/TrainSet"
# duanlie_path = "../../data_3subimg/TrainSet/断裂"
# feiduanlie_path = "../../data_3subimg/TrainSet/非断裂"
# csv_path = os.path.join(root_path, "csv")

# if not os.path.isdir(csv_path):
#     os.makedirs(csv_path)

# for group in ["左侧","右侧","双侧"]:
#     group_dir = os.path.join(duanlie_path, group)
#     duanlie_list = os.listdir(group_dir)
#     for file_name in duanlie_list:
#         img_path = os.path.join(group_dir, file_name)[3:]
#         if img_path[:-9]+img_path[-4:] not in img_paths:
#             img_paths.append(img_path[:-9]+img_path[-4:])
#             labels.append("断裂")

# feiduanlie_list = os.listdir(feiduanlie_path)
# for file_name in feiduanlie_list:
#     img_path = os.path.join(feiduanlie_path, file_name)[3:]
#     if img_path[:-9]+img_path[-4:] not in img_paths:
#         img_paths.append(img_path[:-9]+img_path[-4:])
#         labels.append("非断裂")


# side
side_path = "../../data_3subimg/TrainSet/断裂"
csv_path = os.path.join(side_path, "csv")
if not os.path.isdir(csv_path):
    os.makedirs(csv_path)

for category in ["左侧","右侧","双侧"]:
    class_dir = os.path.join(side_path, category)
    file_list = os.listdir(class_dir)
    for file_name in file_list:
        img_path = os.path.join(class_dir, file_name)[3:]
        if img_path[:-9]+img_path[-4:] not in img_paths:
            img_paths.append(img_path[:-9]+img_path[-4:])
            labels.append(category)


img_paths = np.array(img_paths)
labels = np.array(labels)
for k, (train_idx, val_idx) in enumerate(kf.split(img_paths, labels)):
    train_path = img_paths[train_idx]
    val_path = img_paths[val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

    train_df.insert(loc=len(train_df.columns), column="path", value=train_path)
    train_df.insert(loc=len(train_df.columns), column="label", value=train_labels)
    train_df.to_csv(csv_path+f"/train_fold{k + 1}.csv", index=False)
    val_df.insert(loc=len(val_df.columns), column="path", value=val_path)
    val_df.insert(loc=len(val_df.columns), column="label", value=val_labels)
    val_df.to_csv(csv_path+f"/val_fold{k + 1}.csv", index=False)
