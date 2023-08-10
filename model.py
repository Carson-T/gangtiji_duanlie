import torch
import torch.nn as nn


class Resnet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Resnet, self).__init__()
        self.model = backbone
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def get_head(self):
        return self.model.fc



class Efficientnet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Efficientnet, self).__init__()
        self.model = backbone
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def get_head(self):
        return self.model.classifier

class Convnext(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Convnext, self).__init__()
        self.model = backbone
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

    def get_head(self):
        return self.model.head

class Convnext_base_tv(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Convnext_base_tv, self).__init__()
        self.model = backbone
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier)[:-1],
            nn.Linear(self.model.head.fc.in_features, num_classes)
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def get_head(self):
        return self.model.classifier
