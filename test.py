import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
# from dataset import *
# from model import *


class TestDataset(Dataset):
    def __init__(self, testpath, transform):
        super(TestDataset, self).__init__()
        self.testpath = testpath
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
        return img_list, label, img_path

    def _make_dataset(self):
        img_paths = []
        labels = []
        for group in self.groups:
            group_path = os.path.join(self.testpath, group)
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


class SideTestDataset(Dataset):
    def __init__(self, test_path, transform):
        super(SideTestDataset, self).__init__()
        self.test_path = test_path
        self.class_dict = {"左侧": 0, "右侧": 1, "双侧": 2}
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
        return img_list, label, img_path

    def _make_dataset(self):
        img_paths = []
        labels = []
        for group in self.groups:
            group_path = os.path.join(self.test_path, os.path.join(group, "断裂"))
            for class_name in self.class_dict:
                class_path = os.path.join(group_path, class_name)
                label = self.class_dict[class_name]
                for file_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, file_name)
                    if img_path[:-9] + img_path[-4:] not in img_paths:
                        img_paths.append(img_path[:-9] + img_path[-4:])
                        labels.append(label)

        return img_paths, labels

class Convnext(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Convnext, self).__init__()
        self.branch1 = backbone
        self.branch2 = copy.deepcopy(backbone)
        self.branch3 = copy.deepcopy(backbone)
        self.branchs = nn.ModuleList([self.branch1, self.branch2, self.branch3])
        self.classifier = nn.Sequential(
            nn.Linear(backbone.head.fc.in_features*3, num_classes),
        )

    def forward(self, x):
        for i in range(len(x)):
            features = self.branchs[i].forward_features(x[i])
            pre_logits = self.branchs[i].forward_head(features, pre_logits=True)
            if i == 0:
                output = pre_logits
            else:
                output = torch.hstack([output, pre_logits])
        output = self.classifier(output)
        return output

    def get_head(self):
        return self.classifier

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
        count_dict = collections.Counter(all_preds[:][i])
        voted_preds.append(count_dict.most_common(1)[0][0])
    voted_preds = torch.tensor(voted_preds)
    return voted_preds


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mode = "duanlie"   # side or duanlie
    testpath = "../data_3subimg/TestSet"
    model_paths = [     # duanlie
        "convnextv2_n-3subimg-novalid-v7.pth"
    ]
    # model_paths = [   # side
    #     "convnextv2_n-3subimg-side-v2.pth"
    # ]

    test_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    if mode == "duanlie":
        test_loader = DataLoader(TestDataset(testpath, test_transform), batch_size=32, shuffle=False,
                                 num_workers=8, pin_memory=True, drop_last=False)
        model1 = Convnext(timm.create_model("convnextv2_nano.fcmae_ft_in1k"), 2).to(device)
    elif mode == "side":
        test_loader = DataLoader(SideTestDataset(testpath, test_transform), batch_size=32,
                                 shuffle=False,
                                 num_workers=8, pin_memory=True, drop_last=False)
        model1 = Convnext(timm.create_model("convnextv2_nano.fcmae_ft_in1k"), 3).to(device)
    # model1 = Efficientnet(timm.create_model("efficientnetv2_rw_s.ra2_in1k"), 2).to(device)
    models = [model1] * 5

    all_outputs = []
    all_preds = []
    all_acc = []
    all_auc = []
    for i in range(len(model_paths)):
        load_model(models[i], model_paths[i], device)
        outputs, targets, img_paths = test(test_loader, models[i], device)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).sum().item() / len(targets)
        if mode == "duanlie":
            auc = roc_auc_score(targets, outputs[:, 1])
            all_auc.append(auc)
            all_outputs.append(outputs)
        all_preds.append(preds)
        all_acc.append(acc)

    if mode == "duanlie":
        if len(model_paths) > 1:
            average_outputs = sum(all_outputs) / len(all_outputs)
            average_auc = roc_auc_score(targets, average_outputs[:, 1])
            # vote
            voted_preds = vote(targets, all_preds)
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

    elif mode == "side":
        if len(model_paths) > 1:
            voted_preds = vote(targets, all_preds)
            average_acc = (voted_preds == targets).sum().item() / len(targets)
        else:
            voted_preds = all_preds[0]
            average_acc = all_acc[0]
        print("all_acc:", all_acc)
        print("average_acc:", average_acc)

    # plot_matrix(targets, voted_preds, [0, 1],
    #                     "jingxi_confusion_matrix.jpg",
    #                     ['standards', 'non-standards'])

    # probility, _ = torch.max(average_outputs, 1)
    # df = pd.DataFrame({"img_path": img_paths, "prob": probility.tolist(), "pred": voted_preds, "label": targets})
    # df.to_csv("../jingxi_predict.csv", index=False, encoding="gbk")
