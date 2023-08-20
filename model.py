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
        self.models = [backbone]*3
        self.classifier = nn.Sequential(
            nn.Linear(backbone.head.fc.in_features*3, num_classes),
        )

    def forward(self, x):
        output = torch.tensor([])
        for i in range(len(x)):
            pre_logits = self.models[i].forward_features(x[i])
            pre_logits = self.models[i].forward_head(pre_logits, pre_logits=True)
            output = torch.hstack([output, pre_logits])
        output = self.classifier(output)
        return output

    def get_head(self):
        return self.classifier

