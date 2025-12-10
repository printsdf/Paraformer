import torch
import torch.nn as nn
import torch.nn.functional as F

class SmoothCE(nn.Module):
    def __init__(self, eps=0.1, ignore_index=255):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        n, c, h, w = logits.shape
        log_prob = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob).scatter_(1, target.unsqueeze(1), 1)
            true_dist = true_dist * (1 - self.eps) + self.eps / c
            true_dist[target == self.ignore_index] = 0
        loss = -(true_dist * log_prob).sum(dim=1)
        valid = (target != self.ignore_index).float()
        return (loss * valid).sum() / (valid.sum() + 1e-6)

class DiceLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        target = target.clone()
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return logits.new_tensor(0.0)
        target[~mask] = 0
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        probs = probs * mask.unsqueeze(1)
        one_hot = one_hot * mask.unsqueeze(1)
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class ComboLoss(nn.Module):
    def __init__(self, ce_eps=0.1, dice_w=0.5, ignore_index=255):
        super().__init__()
        self.ce = SmoothCE(eps=ce_eps, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_w = dice_w

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_w * self.dice(logits, target)

def make_pseudo_label(logits, hard_target, conf_th=0.7, ignore_index=255):
    with torch.no_grad():
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        pseudo = hard_target.clone()
        pseudo[conf < conf_th] = ignore_index
        return pseudo
