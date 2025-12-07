"""
Custom loss functions for the Paraformer framework.
This module provides ComboLoss (CE + Label Smoothing + Dice) and pseudo label generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ComboLoss(nn.Module):
    """
    Combination loss: Cross-Entropy with Label Smoothing + Dice Loss
    
    Args:
        ce_eps: Label smoothing factor for cross-entropy (default: 0.1)
        dice_w: Weight for dice loss component (default: 0.5)
        ignore_index: Index to ignore in loss calculation (default: 255)
    """
    def __init__(self, ce_eps=0.1, dice_w=0.5, ignore_index=255):
        super(ComboLoss, self).__init__()
        self.ce_eps = ce_eps
        self.dice_w = dice_w
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, label_smoothing=ce_eps)
    
    def dice_loss(self, pred, target, num_classes):
        """
        Compute Dice loss
        
        Args:
            pred: Predictions (logits), shape [B, C, H, W]
            target: Ground truth labels, shape [B, H, W]
            num_classes: Number of classes
        
        Returns:
            Dice loss value
        """
        pred = F.softmax(pred, dim=1)
        
        # Create mask for valid pixels (not ignore_index) before one_hot encoding
        mask = (target != self.ignore_index).float().unsqueeze(1)
        
        # Clamp target to valid range [0, num_classes-1] for one_hot encoding
        # Invalid pixels will be masked out anyway
        target_clamped = target.clamp(0, num_classes - 1)
        target_one_hot = F.one_hot(target_clamped, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Apply mask to both pred and target_one_hot
        pred = pred * mask
        target_one_hot = target_one_hot * mask
        
        # Compute Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice_coef = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1. - dice_coef.mean()
        
        return dice_loss
    
    def forward(self, pred, target):
        """
        Compute combined loss
        
        Args:
            pred: Predictions (logits), shape [B, C, H, W]
            target: Ground truth labels, shape [B, H, W]
        
        Returns:
            Combined loss value
        """
        ce = self.ce_loss(pred, target)
        
        # Only compute dice loss if there are valid pixels
        if (target != self.ignore_index).any():
            num_classes = pred.size(1)
            dice = self.dice_loss(pred, target, num_classes)
            return ce + self.dice_w * dice
        else:
            return ce


def make_pseudo_label(pred_logits, true_labels, conf_th=0.7, ignore_index=255):
    """
    Generate pseudo labels with confidence filtering
    
    This function creates pseudo labels by:
    1. Using true labels where available
    2. Using high-confidence predictions where true labels are ignore_index
    
    Args:
        pred_logits: Model predictions (logits), shape [B, C, H, W]
        true_labels: Ground truth labels, shape [B, H, W]
        conf_th: Confidence threshold for pseudo labels (default: 0.7)
        ignore_index: Index representing unlabeled/ignore pixels (default: 255)
    
    Returns:
        Pseudo labels, shape [B, H, W]
    """
    with torch.no_grad():
        # Get prediction probabilities and predicted classes
        probs = F.softmax(pred_logits, dim=1)
        max_probs, pred_classes = torch.max(probs, dim=1)
        
        # Start with true labels
        pseudo = true_labels.clone()
        
        # For pixels with ignore_index in true labels, use high-confidence predictions
        ignore_mask = (true_labels == ignore_index)
        high_conf_mask = (max_probs > conf_th) & ignore_mask
        
        # Replace ignore_index pixels with high-confidence predictions
        pseudo[high_conf_mask] = pred_classes[high_conf_mask]
        
        return pseudo
