import torch
import torch.nn as nn
import timm
from timm.models import register_model
from safetensors.torch import load_file
import copy

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

@register_model
def MyConvnext(backbone, pretrained_path, num_classes, is_pretrained=False, **kwargs) -> Convnext:
    pretrained_model = timm.create_model(backbone, drop_rate=kwargs["drop_rate"], drop_path_rate=kwargs["drop_path_rate"], pretrained=is_pretrained)
    if pretrained_path:
        pretrained_model.load_state_dict(load_file(pretrained_path))
    model = Convnext(pretrained_model, num_classes)
    return model