import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import albumentations
from albumentations import pytorch as AT
from torch.utils.data import DataLoader


class Two_Dataset(Dataset):
    def __init__(self,data_path, transform, is_test, is_concat):
        super(Two_Dataset, self).__init__()
        self.data_path = data_path
        self.is_concat = is_concat
        if is_test == False:
            # self.groups = ["1.佛山市医", "2.湖南省妇幼", "3.广医三院", "4.白银", "5.陕西省人民医院"]
            self.groups = ["2.湖南省妇幼", "4.白银", "5.陕西省人民医院"]
            self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
            self.group_paths.append(os.path.join(data_path, "TrainSet"))
        else:
            self.groups = ["1.佛山市医", "3.广医三院"]
            self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
        self.class_dict = {"非断裂": 0, "断裂": 1}  # label dictionary
        self.transform = transform
        self.img_paths, self.labels = self._make_dataset()  # make dataset

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img_list = []
        # img = Image.open(img_path)
        for i in range(3):
            img = cv2.imread(img_path[:-4] + f"-sub{i + 1}" + img_path[-4:])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)

        if self.is_concat:
            img_list = [img[0] for img in img_list]
            img_list = torch.stack(img_list)
        return img_list, label

    def _make_dataset(self):
        img_paths = []
        labels = []
        for group_path in self.group_paths:
            for class_name in self.class_dict:
                class_path = os.path.join(group_path, class_name)
                label = self.class_dict[class_name]
                if class_name == "断裂":
                    for ce in ["左侧", "右侧", "双侧"]:
                        ce_path = os.path.join(class_path, ce)
                        for file_name in os.listdir(ce_path):
                            img_path = os.path.join(ce_path, file_name)
                            if img_path[:-9] + img_path[-4:] not in img_paths:
                                img_paths.append(img_path[:-9] + img_path[-4:])
                                labels.append(label)

                else:
                    for file_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, file_name)
                        if img_path[:-9] + img_path[-4:] not in img_paths:
                            img_paths.append(img_path[:-9] + img_path[-4:])
                            labels.append(label)

        return img_paths, labels


class Three_Dataset(Dataset):
    def __init__(self, data_path, transform, is_test, is_concat):
        super(Three_Dataset, self).__init__()
        self.data_path = data_path
        self.is_concat = is_concat
        if is_test == False:
            self.groups = ["2.湖南省妇幼", "4.白银", "5.陕西省人民医院"]
            self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
            self.group_paths.append(os.path.join(data_path, "TrainSet"))
        else:
            self.groups = ["1.佛山市医", "3.广医三院"]
            self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
        self.class_dict = {"左侧": 0, "右侧": 1, "双侧": 2}
        self.transform = transform
        self.img_paths, self.labels = self._make_dataset()  # make dataset

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img_list = []
        # img = Image.open(img_path)
        for i in range(3):
            img = cv2.imread(img_path[:-4] + f"-sub{i + 1}" + img_path[-4:])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)

        if self.is_concat:
            img_list = [img[0] for img in img_list]
            img_list = torch.stack(img_list)
        return img_list, label

    def _make_dataset(self):
        img_paths = []
        labels = []
        for group_path in self.group_paths:
            group_path = os.path.join(group_path, "断裂")
            for class_name in self.class_dict:
                class_path = os.path.join(group_path, class_name)
                label = self.class_dict[class_name]
                for file_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, file_name)
                    if img_path[:-9] + img_path[-4:] not in img_paths:
                        img_paths.append(img_path[:-9] + img_path[-4:])
                        labels.append(label)

        return img_paths, labels

class Four_Dataset(Dataset):
    def __init__(self, data_path, transform, is_test, is_concat):
        super(Four_Dataset, self).__init__()
        self.data_path = data_path
        self.is_concat = is_concat
        if is_test == False:
            # self.groups = ["1.佛山市医", "2.湖南省妇幼", "3.广医三院", "4.白银", "5.陕西省人民医院"]
            self.groups = ["2.湖南省妇幼", "4.白银", "5.陕西省人民医院"]
            self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
            self.group_paths.append(os.path.join(data_path, "TrainSet"))
        else:
            self.groups = ["1.佛山市医", "3.广医三院"]
            self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
        self.class_dict = {"非断裂": 0, "左侧": 1, "右侧": 2, "双侧": 3}  # label dictionary
        self.transform = transform
        self.img_paths, self.labels = self._make_dataset()  # make dataset

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img_list = []
        # img = Image.open(img_path)
        for i in range(3):
            img = cv2.imread(img_path[:-4] + f"-sub{i + 1}" + img_path[-4:])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)

        if self.is_concat:
            img_list = [img[0] for img in img_list]
            img_list = torch.stack(img_list)
        return img_list, label

    def _make_dataset(self):
        img_paths = []
        labels = []
        for group_path in self.group_paths:
            for class_name in self.class_dict:
                label = self.class_dict[class_name]
                if class_name in ["左侧", "右侧", "双侧"]:
                    class_path = os.path.join(group_path, "断裂/"+class_name)
                    for file_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, file_name)
                        if img_path[:-9] + img_path[-4:] not in img_paths:
                            img_paths.append(img_path[:-9] + img_path[-4:])
                            labels.append(label)

                else:
                    class_path = os.path.join(group_path, class_name)
                    for file_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, file_name)
                        if img_path[:-9] + img_path[-4:] not in img_paths:
                            img_paths.append(img_path[:-9] + img_path[-4:])
                            labels.append(label)

        return img_paths, labels


# class MultiTask_Dataset(Dataset):
#     def __init__(self, data_path, transform, is_test):
#         super(MultiTask_Dataset, self).__init__()
#         self.data_path = data_path
#         if is_test == False:
#             self.groups = ["2.湖南省妇幼", "4.白银", "5.陕西省人民医院"]
#             self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
#             self.group_paths.append(os.path.join(data_path, "TrainSet"))
#         else:
#             self.groups = ["1.佛山市医", "3.广医三院"]
#             self.group_paths = [os.path.join(data_path+"/TestSet", i) for i in self.groups]
#         self.class_dict = {"左侧": 0, "右侧": 1, "双侧": 2}
#         self.transform = transform
#         self.img_paths, self.labels = self._make_dataset_three()  # make dataset
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         img_path = self.img_paths[idx]
#         label = self.labels[idx]
#         img_list = []
#         # img = Image.open(img_path)
#         for i in range(3):
#             img = cv2.imread(img_path[:-4] + f"-sub{i + 1}" + img_path[-4:])
#             # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             if self.transform:
#                 # img = self.transform(img)
#                 img = self.transform(image=img)["image"]
#             img_list.append(img)
#         return img_list, label
#
#     def _make_dataset_two(self):
#         img_paths = []
#         labels = []
#         for group_path in self.group_paths:
#             for class_name in self.class_dict:
#                 class_path = os.path.join(group_path, class_name)
#                 label = self.class_dict[class_name]
#                 if class_name == "断裂":
#                     for ce in ["左侧", "右侧", "双侧"]:
#                         ce_path = os.path.join(class_path, ce)
#                         for file_name in os.listdir(ce_path):
#                             img_path = os.path.join(ce_path, file_name)
#                             if img_path[:-9] + img_path[-4:] not in img_paths:
#                                 img_paths.append(img_path[:-9] + img_path[-4:])
#                                 labels.append(label)
#
#                 else:
#                     for file_name in os.listdir(class_path):
#                         img_path = os.path.join(class_path, file_name)
#                         if img_path[:-9] + img_path[-4:] not in img_paths:
#                             img_paths.append(img_path[:-9] + img_path[-4:])
#                             labels.append(label)
#
#         return img_paths, labels
#     def _make_dataset_three(self):
#         img_paths = []
#         labels = []
#         for group_path in self.group_paths:
#             group_path = os.path.join(group_path, "断裂")
#             for class_name in self.class_dict:
#                 class_path = os.path.join(group_path, class_name)
#                 label = self.class_dict[class_name]
#                 for file_name in os.listdir(class_path):
#                     img_path = os.path.join(class_path, file_name)
#                     if img_path[:-9] + img_path[-4:] not in img_paths:
#                         img_paths.append(img_path[:-9] + img_path[-4:])
#                         labels.append(label)
#
#         return img_paths, labels
#
#
#


if __name__ == '__main__':
    train_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(),
        albumentations.ToGray(p=1),
        AT.ToTensorV2()
    ])
    loader = DataLoader(Two_Dataset("../data_3subimg", train_transform, is_test=False, is_concat=True),
                              batch_size=4, shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)

    for (i, j) in loader:
        print(i)
        print(len(i))
        print(i.shape)
        # print(i.type)
        print(j)
        break

