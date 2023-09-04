import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import albumentations
from albumentations import pytorch as AT
from torch.utils.data import DataLoader


class TrainValDataset(Dataset):
    def __init__(self, csv_path, transform):
        super(TrainValDataset, self).__init__()
        self.csv_path = csv_path
        self.class_dict = {"断裂": 0, "非断裂": 1}  # label dictionary
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
            # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)
        return img_list, label

    def _make_dataset(self):
        data = pd.read_csv(self.csv_path)
        img_paths = data["path"].values.tolist()
        labels = [self.class_dict[i] for i in data["label"].values]

        return img_paths, labels


class NoValDataset(Dataset):
    def __init__(self, train_path, transform):
        super(NoValDataset, self).__init__()
        self.train_path = train_path
        self.class_dict = {"断裂": 0, "非断裂": 1}
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
            # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)
        return img_list, label

    def _make_dataset(self):
        img_paths = []
        labels = []
        for class_name in self.class_dict:
            class_path = os.path.join(self.train_path, class_name)
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


class TestDataset(Dataset):
    def __init__(self, test_path, transform):
        super(TestDataset, self).__init__()
        self.test_path = test_path
        self.class_dict = {"断裂": 0, "非断裂": 1}
        self.groups = ["1.佛山市医", "2.湖南省妇幼", "3.广医三院", "4.白银", "5.陕西省人民医院"]
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
            # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)
        return img_list, label

    def _make_dataset(self):
        img_paths = []
        labels = []
        for group in self.groups:
            group_path = os.path.join(self.test_path, group)
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


if __name__ == '__main__':
    train_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])
    train_loader = DataLoader(TrainValDataset("../data_3subimg/TrainSet/csv/train_fold1.csv", train_transform),
                              batch_size=4, shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)

    for (i, j) in train_loader:
        print(i)
        print(len(i))
        print(i[0].shape)
        # print(i.type)
        print(j)
        break

