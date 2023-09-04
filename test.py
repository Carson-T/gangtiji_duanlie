import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm          # version >= 0.92
import albumentations
from albumentations import pytorch as AT
import ttach as tta
from sklearn.metrics import roc_auc_score
import cv2
import os
import pandas as pd
import collections
from tqdm import tqdm
import copy
from model import *

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

    return all_outputs.cpu().detach(), all_targets.cpu().detach()


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

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(),
        AT.ToTensorV2()
    ])

    test_loader = DataLoader(TestDataset("../data_3subimg/TestSet", test_transform), batch_size=32, shuffle=False, num_workers=8, 
                            pin_memory=True, drop_last=False)

    model1 = Convnext(timm.create_model("convnextv2_nano.fcmae_ft_in1k"), 2).to(device)
    # model1 = Efficientnet(timm.create_model("efficientnetv2_rw_s.ra2_in1k"), 2).to(device)
    models = [model1]*5

    model_paths = [                
        "../saved_model/convnext/convnextv2_n-3subimg-fold1-v5.pth",
        "../saved_model/convnext/convnextv2_n-3subimg-fold2-v5.pth",
        "../saved_model/convnext/convnextv2_n-3subimg-fold3-v5.pth",
        "../saved_model/convnext/convnextv2_n-3subimg-fold4-v5.pth",
        "../saved_model/convnext/convnextv2_n-3subimg-fold5-v5.pth"
        ]
    # model_paths = [
    #     "../saved_model/efficientnet/efficientnetv2_s-3subimg-fold1-v3.pth",
    #     "../saved_model/efficientnet/efficientnetv2_s-3subimg-fold2-v3.pth",
    #     "../saved_model/efficientnet/efficientnetv2_s-3subimg-fold3-v3.pth",
    #     "../saved_model/efficientnet/efficientnetv2_s-3subimg-fold4-v3.pth",
    #     "../saved_model/efficientnet/efficientnetv2_s-3subimg-fold5-v3.pth",
    # ]



    # tta_transforms = tta.Compose(    #Test Time Augmentation
    #     [
    #         # tta.HorizontalFlip(),
    #         # tta.Rotate90(angles=[0, 180]),
    #         tta.Scale(scales=[0.8, 1, 1.2]),
    #         # tta.Multiply(factors=[0.9, 1, 1.1]),
    #     ]
    # )

    all_outputs = []
    all_preds = []
    all_acc = []
    all_auc = []
    for i in range(len(model_paths)):
        load_model(models[i], model_paths[i], device)
        # tta_model = tta.ClassificationTTAWrapper(models[i], tta_transforms)
        outputs, targets = test(test_loader, models[i], device)
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

