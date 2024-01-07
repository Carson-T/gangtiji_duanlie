import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import timm  # version >= 0.92
import albumentations
from albumentations import pytorch as AT
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import ttach as tta
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score
import cv2
import os
import pandas as pd
import collections
from tqdm import tqdm
import copy
from models import *

class TestDataset(Dataset):
    def __init__(self, data_path, transform, is_four):
        super(TestDataset, self).__init__()
        self.data_path = data_path
        self.is_four = is_four
        self.two_class_dict = {"非断裂": 0, "断裂": 1}
        self.four_class_dict = {"非断裂": 0, "左侧": 1, "右侧": 2, "双侧": 3}
        self.transform = transform
        self.img_paths, self.duanlie_labels, self.four_labels = self._make_dataset()  # make dataset

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        if self.is_four:
            label = self.four_labels[idx]
        else:
            label = self.duanlie_labels[idx]
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
        data = pd.read_excel(self.data_path)
        img_paths = data["img_path"].values
        duanlie_labels = data["two_label"].values
        side_labels = data["four_label"].values

        return img_paths, duanlie_labels, side_labels

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
    testpath = "../shuffle_testdata_label.xlsx"
    model_paths = [  # duanlie
        "../saved_model/efficientnet/efficientnetv2_s-3subimg-new_data-duanlie-v2-fold1.pth",

    ]

    test_loader = DataLoader(TestDataset(testpath, test_transform, is_four=False), batch_size=32, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)

    model = timm.create_model(model_name="MyEfficientnet_tv",
                              num_classes=2,
                              drop_rate=0,
                              drop_path_rate=0
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
        {"img_path": img_paths, "two_label": targets, "two_preds": voted_preds, "two_output0": average_outputs[:, 0].tolist(),
         "two_output1": average_outputs[:, 1].tolist()})
    # df.to_excel("./shuffle_testdata_label.xlsx", index=True)
    return df

def side_inference(test_transform):
    testpath = "../shuffle_testdata_label.xlsx"
    model_paths = [
        "../saved_model/efficientnet/efficientnetv2_s-3subimg-new_data-side-v4-fold1.pth",

    ]

    test_loader = DataLoader(TestDataset(testpath, test_transform, is_four=True), batch_size=32, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)

    model = timm.create_model(model_name="MyEfficientnet_tv",
                              num_classes=3,
                              drop_rate=0,
                              drop_path_rate=0
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
        # acc = (preds == targets).sum().item() / len(targets)
        all_outputs.append(outputs)
        all_preds.append(preds.numpy().tolist())
        # all_acc.append(acc)

    # if len(model_paths) > 1:
    #     average_outputs = sum(all_outputs) / len(all_outputs)
    #     # vote
    #     voted_preds = vote(targets, all_preds)
    #     # voted_preds = torch.argmax(average_outputs, dim=1)
    #     average_acc = (voted_preds == targets).sum().item() / len(targets)
    # else:
    average_outputs = all_outputs[0]
    voted_preds = all_preds[0]
    # average_acc = all_acc[0]

    # print("all_acc:", all_acc)
    # print("average_acc:", average_acc)
    df = pd.DataFrame(
        {"img_path": img_paths, "four_label": targets, "four_preds": voted_preds, "side_output0": average_outputs[:, 0].tolist(),
         "side_output1": average_outputs[:, 1].tolist(), "side_output2": average_outputs[:, 2].tolist()})
    # df.to_excel("./shuffle_testdata_label.xlsx", index=True)
    return df

def four_inference(test_transform):
    testpath = "../shuffle_testdata_label.xlsx"
    model_paths = [
        "../saved_model/efficientnet/efficientnetv2_s-3subimg-new_data-four-v2-fold1.pth",
    ]
    test_loader = DataLoader(TestDataset(testpath, test_transform, is_four=True), batch_size=32, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)

    model = timm.create_model(model_name="MyEfficientnet",
                              backbone="efficientnetv2_rw_s.ra2_in1k",
                              pretrained_path=False,
                              num_classes=4,
                              drop_rate=0,
                              drop_path_rate=0
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

    # if len(model_paths) > 1:
    #     average_outputs = sum(all_outputs) / len(all_outputs)
    #     # vote
    #     voted_preds = vote(targets, all_preds)
    #     # voted_preds = torch.argmax(average_outputs, dim=1)
    #     average_acc = (voted_preds == targets).sum().item() / len(targets)
    # else:
    average_outputs = all_outputs[0]
    voted_preds = all_preds[0]
    average_acc = all_acc[0]

    print("all_acc:", all_acc)
    print("average_acc:", average_acc)
    print("recall:", recall_score(targets, voted_preds, average="micro"))
    print("precision:", precision_score(targets, voted_preds, average="micro"))
    df = pd.DataFrame(
        {"img_path": img_paths, "four_label": targets, "preds": voted_preds, "four_output0": average_outputs[:, 0].tolist(),
         "four_output1": average_outputs[:, 1].tolist(), "four_output2": average_outputs[:, 2].tolist(), "four_output3": average_outputs[:, 3].tolist(),})
    # df.to_excel("./shuffle_testdata_label.xlsx", index=True)
    return df

def grad_cam(test_transform):
    testpath = "../shuffle_testdata_label.xlsx"
    model_paths = [
        "../saved_model/efficientnet/efficientnetv2_s-3subimg-new_data-four-v2-fold1.pth",
    ]
    test_loader = DataLoader(TestDataset(testpath, test_transform, is_four=True), batch_size=1, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=False)

    model = timm.create_model(model_name="MyEfficientnet",
                              backbone="efficientnetv2_rw_s.ra2_in1k",
                              pretrained_path=False,
                              num_classes=4,
                              drop_rate=0,
                              drop_path_rate=0
                              )
    model.to(device)
    load_model(model, model_paths[0], device)

    cam1 = GradCAM(model.branch1, target_layers=[model.branch1.conv_head], use_cuda=True)
    cam2 = GradCAM(model.branch2, target_layers=[model.branch2.conv_head], use_cuda=True)
    cam3 = GradCAM(model.branch3, target_layers=[model.branch3.conv_head], use_cuda=True)

    model.eval()
    for i, (img_list, targets, img_paths) in enumerate(tqdm(test_loader)):
        img_list = [img.to(device) for img in img_list]
        for j in range(len(img_list)):
            if j == 0:
                grayscale_cam = cam1(input_tensor=img_list[j], targets=None)
            elif j == 1:
                grayscale_cam = cam2(input_tensor=img_list[j], targets=None)
            elif j == 2:
                grayscale_cam = cam3(input_tensor=img_list[j], targets=None)
            grayscale_cam = grayscale_cam[0, :]
            img = cv2.imread(img_paths[0][:-4] + f"-sub{j+1}" + img_paths[0][-4:], 1)
            img = cv2.resize(img, (224, 224)) / 255
            # print(img)
            visualization = show_cam_on_image(img, grayscale_cam)
            # print(visualization)
            cv2.imwrite("../shuffle_testdata_cam/"+str(i)+f"-sub{j+1}"+img_paths[0][-4:], visualization)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])
    # grad_cam(test_transform)

    two_df = duanlie_inference(test_transform)
    print(two_df)
    side_df = side_inference(test_transform)
    print(side_df)
    # result = pd.merge(two_df, side_df, how="left", on=["img_path"])
    # for i in range(len(result)):
    #     if result.loc[i, "two_preds"] == 0:
    #         result.loc[i, "four_preds"] = 0
    #     else:
    #         result.loc[i, "four_preds"] += 1
    #
    # four_df = four_inference(test_transform)
    # shuffle_testdata_label = pd.read_excel("../shuffle_testdata_label.xlsx")

    # two_three_result = pd.merge(shuffle_testdata_label, result, how="left", on=["img_path", "two_label", "four_label"])
    # two_three_result.to_excel("../two_three_result.xlsx", index=False)

    # four_result = pd.merge(shuffle_testdata_label, four_df, how="left", on=["img_path", "four_label"])
    # four_result.to_excel("../four_result.xlsx", index=False)


