import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import cv2
import ttach as tta
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
import albumentations
from torch.utils.data import DataLoader
from albumentations import pytorch as AT
from tqdm import tqdm
from model import *
from dataset import TestDataset
import collections


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


def test(test_loader, model, device):
    model.eval()
    # test_loss = 0.0
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            targets = targets.to(device)
            output = model(images)
            # loss = criterion(output, targets)
            # test_loss += loss.item()
            if i == 0:
                all_outputs = output
                all_targets = targets
            else:
                all_outputs = torch.cat((all_outputs, output))
                all_targets = torch.cat((all_targets, targets))
    all_outputs = F.softmax(all_outputs, dim=1)

    return all_outputs.cpu().detach(), all_targets.cpu().detach()


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform1 = albumentations.Compose([
        albumentations.Resize(224,442),
        albumentations.Normalize(0.21162076, 0.22596906),
        AT.ToTensorV2()
    ])

    test_transform2 = albumentations.Compose([
        albumentations.Resize(224,442),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    test_loader1 = DataLoader(TestDataset("../data/TestSet", test_transform1, "J"),
                             batch_size=16, shuffle=False, num_workers=8, drop_last=False)

    test_loader2 = DataLoader(TestDataset("../data/TestSet", test_transform2, "J"),
                             batch_size=16, shuffle=False, num_workers=8, drop_last=False)                  

    model1 = myconvnext(timm.create_model("convnextv2_nano.fcmae_ft_in1k"), 2).to(device)
    model2 = resnet(timm.create_model("resnet50.tv_in1k"), 2).to(device)
    model3 = efficientnet(timm.create_model("efficientnetv2_rw_s.ra2_in1k"), 2).to(device)
    models = [model1, model2, model3]

    model_paths = [
        "../saved_model/J/convnextv2_n-J-fold3-v2.pth"
        # "../saved_model/J/resnet/resnet50-J-fold1-v1.pth",
        # "../saved_model/J/efficientnet/efficientnetv2_s-J-fold1-v1.pth"
        ]

    tta_transforms = tta.Compose(
        [
            # tta.HorizontalFlip(),
            # tta.Rotate90(angles=[0, 180]),
            tta.Scale(scales=[1, 2, 4]),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )




    # load model

    all_outputs = []
    all_preds = []
    all_acc = []
    all_auc = []
    for i in range(len(model_paths)):
        load_model(models[i], model_paths[i], device)
        tta_model = tta.ClassificationTTAWrapper(models[i], tta_transforms)
        outputs, targets = test(test_loader1 if i==0 else test_loader2, models[i], device)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).sum().item() / len(targets)
        auc = roc_auc_score(targets, outputs[:, 1])
        all_outputs.append(outputs)
        all_preds.append(preds)
        all_acc.append(acc)
        all_auc.append(auc)


    print(all_acc)
    print(all_auc)

    average_outputs = sum(all_outputs)/len(targets)
    average_auc = roc_auc_score(targets, average_outputs[:, 1])
    print(average_auc)

