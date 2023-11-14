import torch
import torch.nn as nn
import timm
from timm.models import register_model
from safetensors.torch import load_file
import copy


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

class Efficientnet_gray(nn.Module):
    def __init__(self, backbone, num_classes):
        super(Efficientnet_gray, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(backbone.classifier.in_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pre_logits = self.backbone.forward_head(features, pre_logits=True)
        output = self.classifier(pre_logits)
        return output

    def get_head(self):
        return self.classifier

@register_model
def MyEfficientnet(backbone, pretrained_path, num_classes, is_pretrained=False, **kwargs) -> Efficientnet:
    pretrained_model = timm.create_model(backbone, drop_rate=kwargs["drop_rate"], drop_path_rate=kwargs["drop_path_rate"], pretrained=is_pretrained)
    if pretrained_path:
        pretrained_model.load_state_dict(load_file(pretrained_path))
    model = Efficientnet(pretrained_model, num_classes)
    return model

@register_model
def MyEfficientnet_gray(backbone, pretrained_path, num_classes, is_pretrained=False, **kwargs) -> Efficientnet_gray:
    pretrained_model = timm.create_model(backbone, drop_rate=kwargs["drop_rate"], drop_path_rate=kwargs["drop_path_rate"], pretrained=is_pretrained)
    if pretrained_path:
        pretrained_model.load_state_dict(load_file(pretrained_path))
    model = Efficientnet_gray(pretrained_model, num_classes)
    return model