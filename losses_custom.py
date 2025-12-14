import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCE(nn.Module):
    """带标签平滑的交叉熵损失"""
    def __init__(self, eps=0.1, ignore_index=255):
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        n, c, h, w = logits.shape
        log_prob = F.log_softmax(logits, dim=1)
        
        # 创建有效像素的 mask
        valid_mask = (target != self.ignore_index)
        
        # 将 ignore_index 位置临时替换为 0，避免 scatter_ 越界
        target_safe = target.clone()
        target_safe[~valid_mask] = 0
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_prob).scatter_(1, target_safe.unsqueeze(1), 1)
            true_dist = true_dist * (1 - self.eps) + self.eps / c
        
        loss = -(true_dist * log_prob).sum(dim=1)
        
        # 只对有效像素计算损失
        valid_mask_float = valid_mask.float()
        return (loss * valid_mask_float).sum() / (valid_mask_float.sum() + 1e-6)


class DiceLoss(nn.Module):
    """Dice损失，适用于分割任务"""
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        
        target_clone = target.clone()
        mask = (target_clone != self.ignore_index)
        
        if mask.sum() == 0:
            return logits.new_tensor(0.0)
        
        # 将 ignore_index 位置临时替换为 0，避免 one_hot 越界
        target_clone[~mask] = 0
        
        # 确保 target 值在有效范围内
        target_clone = target_clone.clamp(0, num_classes - 1)
        
        one_hot = F.one_hot(target_clone, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 扩展 mask 到 [N, C, H, W]
        mask_expanded = mask.unsqueeze(1).expand_as(probs)
        probs = probs * mask_expanded
        one_hot = one_hot * mask_expanded
        
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    """组合损失：SmoothCE + Dice"""
    def __init__(self, ce_eps=0.1, dice_w=0.5, ignore_index=255):
        super().__init__()
        self.ce = SmoothCE(eps=ce_eps, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_w = dice_w

    def forward(self, logits, target):
        return self.ce(logits, target) + self.dice_w * self.dice(logits, target)


def make_pseudo_label(logits, hard_target, conf_th=0.7, ignore_index=255):
    """生成置信过滤伪标签：低于阈值的预测标记为ignore_index"""
    with torch.no_grad():
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        pseudo = hard_target.clone()
        pseudo[conf < conf_th] = ignore_index
        return pseudo
