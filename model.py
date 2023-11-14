import torch
import torch.nn as nn
import copy
from timm.models import register_model


class Resnet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Resnet, self).__init__()
        self.branch1 = backbone
        self.branch2 = copy.deepcopy(backbone)
        self.branch3 = copy.deepcopy(backbone)
        self.branchs = nn.ModuleList([self.branch1, self.branch2, self.branch3])
        self.classifier = nn.Sequential(
            nn.Linear(backbone.fc.in_features * 3, num_classes),
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



class Efficientnet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Efficientnet, self).__init__()
        self.branch1 = backbone
        self.branch2 = copy.deepcopy(backbone)
        self.branch3 = copy.deepcopy(backbone)
        self.branchs = nn.ModuleList([self.branch1, self.branch2, self.branch3])
        self.classifier = nn.Sequential(
            nn.Linear(backbone.classifier.in_features * 3, num_classes),
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

class Efficientnet_tv(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Efficientnet_tv, self).__init__()
        self.branch1 = backbone
        self.branch2 = copy.deepcopy(backbone)
        self.branch3 = copy.deepcopy(backbone)
        self.branchs = nn.ModuleList([self.branch1, self.branch2, self.branch3])
        self.classifier = nn.Sequential(
            nn.Linear(backbone.classifier[1].in_features * 3, num_classes),
        )

    def forward(self, x):
        for i in range(len(x)):
            features = self.branchs[i].features(x[i])
            features = self.branchs[i].avgpool(features)
            features = torch.flatten(features, 1)
            pre_logits = self.branchs[i].classifier[0](features)
            # pre_logits = self.branchs[i].forward_head(features, pre_logits=True)
            if i == 0:
                output = pre_logits
            else:
                output = torch.hstack([output, pre_logits])
        output = self.classifier(output)
        return output

    def get_head(self):
        return self.classifier


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

class MultiTask_efficientnet_tv(nn.Module):
    def __init__(self, backbone, num_classes1, num_classes2):
        super(MultiTask_efficientnet_tv, self).__init__()
        self.branch1 = backbone
        self.branch2 = copy.deepcopy(backbone)
        self.branch3 = copy.deepcopy(backbone)
        self.branchs = nn.ModuleList([self.branch1, self.branch2, self.branch3])
        self.classifier1 = nn.Sequential(
            nn.Linear(backbone.classifier[1].in_features * 3, num_classes1),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(backbone.classifier[1].in_features * 3, num_classes2)
        )

    def forward(self, x, flag):
        for i in range(len(x)):
            features = self.branchs[i].features(x[i])
            features = self.branchs[i].avgpool(features)
            features = torch.flatten(features, 1)
            pre_logits = self.branchs[i].classifier[0](features)
            # pre_logits = self.branchs[i].forward_head(features, pre_logits=True)
            if i == 0:
                output = pre_logits
            else:
                output = torch.hstack([output, pre_logits])
        if flag == "duanlie":
            output = self.classifier1(output)
        else:
            output = self.classifier2(output)
        return output

    def get_head(self):
        return self.classifier1, self.classifier2
