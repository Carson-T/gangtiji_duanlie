import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import timm  # version >= 0.92
import albumentations
from albumentations import pytorch as AT
import ttach as tta
from sklearn.metrics import roc_auc_score, confusion_matrix
import cv2
import os
import pandas as pd
import collections
from tqdm import tqdm
import copy
from models import *

class Two_Dataset(Dataset):
    def __init__(self, data_path, transform, is_test):
        super(Two_Dataset, self).__init__()
        self.data_path = data_path
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
            # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)
        return img_list, label, img_path

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
    def __init__(self, data_path, transform, is_test):
        super(Three_Dataset, self).__init__()
        self.data_path = data_path
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
            # img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                # img = self.transform(img)
                img = self.transform(image=img)["image"]
            img_list.append(img)
        return img_list, label, img_path

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

def test(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        for i, (img_list, targets, img_paths) in enumerate(tqdm(test_loader)):
            img_list = [img.to(device) for img in img_list]
            targets = targets.to(device)
            # with autocast():
            output = model(img_list)
            if i == 0:
                all_outputs = output
                all_targets = targets
                all_img_paths = list(img_paths)
            else:
                all_outputs = torch.cat((all_outputs, output))
                all_targets = torch.cat((all_targets, targets))
                all_img_paths.extend(list(img_paths))
    all_outputs = F.softmax(all_outputs, dim=1)

    return all_outputs.cpu().detach(), all_targets.cpu().detach(), all_img_paths


def load_model(model, model_path, device):
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = collections.OrderedDict()
    for name, params in state_dict.items():
        if "module" in name:
            name = name[7:]
            new_state_dict[name] = params
        else:
            new_state_dict = state_dict
            break
    del state_dict
    model.load_state_dict(new_state_dict)


def vote(targets, all_preds):
    voted_preds = []
    for i in range(len(targets)):
        count_dict = collections.Counter([all_preds[j][i] for j in range(len(all_preds))])
        if i == 1:
            print(count_dict)
        voted_preds.append(count_dict.most_common(1)[0][0])
    voted_preds = torch.tensor(voted_preds)
    return voted_preds

def duanlie_inference(test_transform):
    testpath = "../data_3subimg"
    model_paths = [  # duanlie
        "../saved_model/efficientnet/efficientnetv2_s-3subimg-new_data-duanlie-v2-fold1.pth",

    ]

    test_loader = DataLoader(Two_Dataset(testpath, test_transform, is_test=True), batch_size=32, shuffle=True,
                             num_workers=8, pin_memory=True, drop_last=False)

    model = timm.create_model(model_name="MyEfficientnet",
                              backbone="efficientnetv2_rw_s.ra2_in1k",
                              pretrained_path=None,
                              num_classes=2,
                              is_pretrained=False
                              )
    model.to(device)
    models = [model] * 5

    all_outputs = []
    all_preds = []
    all_acc = []
    all_auc = []
    for i in range(len(model_paths)):
        load_model(models[i], model_paths[i], device)
        outputs, targets, img_paths = test(test_loader, models[i], device)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).sum().item() / len(targets)
        auc = roc_auc_score(targets, outputs[:, 1])
        all_auc.append(auc)
        all_outputs.append(outputs)
        all_preds.append(preds.numpy().tolist())
        all_acc.append(acc)

    if len(model_paths) > 1:
        average_outputs = sum(all_outputs) / len(all_outputs)
        average_auc = roc_auc_score(targets, average_outputs[:, 1])
        # vote
        voted_preds = vote(targets, all_preds)
        # voted_preds = torch.argmax(average_outputs, dim=1)
        average_acc = (voted_preds == targets).sum().item() / len(targets)
    else:
        average_outputs = all_outputs[0]
        average_auc = all_auc[0]
        voted_preds = all_preds[0]
        average_acc = all_acc[0]

    cm = confusion_matrix(targets, voted_preds)
    print(cm)
    tp = cm[0][0]
    fp = cm[1][0]
    tn = cm[1][1]
    fn = cm[0][1]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)

    print("all_acc:", all_acc)
    print("all_auc:", all_auc)
    print("recall:", recall)
    print("precision:", precision)
    print("specificity:", specificity)
    print("npv:", npv)
    print("average_acc:", average_acc)
    print("average_auc:", average_auc)
    df = pd.DataFrame(
        {"img_path": img_paths, "duanlie_label": targets, "duanlie_output0": average_outputs[:, 0].tolist(),
         "duanlie_output1": average_outputs[:, 1].tolist()})
    # df.to_excel("./shuffle_testdata_label.xlsx", index=True)
    return df

def side_inference(test_transform):
    testpath = "../data_3subimg"
    model_paths = [
        "../saved_model/efficientnet/efficientnetv2_s-3subimg-new_data-side-v4-fold1.pth",

    ]

    test_loader = DataLoader(Three_Dataset(testpath, test_transform, is_test=True), batch_size=32, shuffle=True,
                             num_workers=8, pin_memory=True, drop_last=False)
    model = timm.create_model(model_name="MyEfficientnet",
                              backbone="efficientnetv2_rw_s.ra2_in1k",
                              pretrained_path=None,
                              num_classes=3,
                              is_pretrained=False
                              )
    model.to(device)
    models = [model] * 5

    all_outputs = []
    all_preds = []
    all_acc = []
    for i in range(len(model_paths)):
        load_model(models[i], model_paths[i], device)
        outputs, targets, img_paths = test(test_loader, models[i], device)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).sum().item() / len(targets)
        all_outputs.append(outputs)
        all_preds.append(preds.numpy().tolist())
        all_acc.append(acc)

    if len(model_paths) > 1:
        average_outputs = sum(all_outputs) / len(all_outputs)
        # vote
        voted_preds = vote(targets, all_preds)
        # voted_preds = torch.argmax(average_outputs, dim=1)
        average_acc = (voted_preds == targets).sum().item() / len(targets)
    else:
        average_outputs = all_outputs[0]
        voted_preds = all_preds[0]
        average_acc = all_acc[0]

    print("all_acc:", all_acc)
    print("average_acc:", average_acc)
    df = pd.DataFrame(
        {"img_path": img_paths, "side_label": targets, "side_output0": average_outputs[:, 0].tolist(),
         "side_output1": average_outputs[:, 1].tolist(), "side_output2": average_outputs[:, 2].tolist()})
    # df.to_excel("./shuffle_testdata_label.xlsx", index=True)
    return df


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    duanlie_df = duanlie_inference(test_transform)
    side_df = side_inference(test_transform)
    result = pd.merge(duanlie_df, side_df, how="left", on=["img_path"])
    origin_result = pd.read_excel("shuffle_testdata_label.xlsx")
    new_result = pd.merge(origin_result, result, how="left", on=["img_path", "duanlie_label", "side_label"])
    # result.to_excel("./shuffle_testdata_label.xlsx", index=True)

