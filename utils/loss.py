import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(FocalLoss, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = torch.tensor(alpha).to(self.device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class LabelSmoothLoss(nn.Module):
    def __init__(self, weight, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, targets):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        smooth_loss = -log_preds.sum(dim=-1).mean() if self.reduction == "mean" else -log_preds.sum(dim=-1).sum()
        nll_loss = F.nll_loss(log_preds, targets, self.weight, reduction=self.reduction)
        return (1 - self.epsilon) * nll_loss + self.epsilon * (smooth_loss / n)

