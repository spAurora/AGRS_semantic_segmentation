# -*- coding: utf-8 -*-

"""
AGRS_semantic_segmentation
Loss Function
损失函数
~~~~~~~~~~~~~~~~
code by wHy
Aerospace Information Research Institute, Chinese Academy of Sciences
wanghaoyu191@mails.ucas.ac.cn
"""
import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import numpy as np

class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, ignore_label=0, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()

        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(output, target)


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):
        if output.dim()>2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1,2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.ce_fn(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

# class DiceLoss2d(nn.Module):
#     """
#     Dice Loss for 2D segmentation tasks
#     """
#     def __init__(self, smooth=1e-6, weight =None):
#         super(DiceLoss2d, self).__init__()
#         self.smooth = smooth

#     def forward(self, output, target):
#         """
#         Forward pass
#         :param output: torch.tensor (NxCxHxW) - model logits
#         :param target: torch.tensor (NxHxW) - ground truth labels
#         :return: scalar Dice loss
#         """

#         # Flatten the output and target
#         output = torch.softmax(output, dim=1)
#         output_flat = output.view(-1)
#         target_flat = target.view(-1)

#         # Calculate intersection and union
#         intersection = (output_flat * target_flat).sum()
#         dice_score = (2. * intersection + self.smooth) / (output_flat.sum() + target_flat.sum() + self.smooth)

#         # Return Dice Loss
#         return 1 - dice_score

class DiceLoss2d(nn.Module):
    """
    Dice Loss for 2D segmentation tasks
    """
    def __init__(self, smooth=1e-6, weight=None):
        super(DiceLoss2d, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxCxHxW) - model logits
        :param target: torch.tensor (NxHxW) - ground truth labels
        :return: scalar Dice loss
        """
        # Ensure the target is of the same shape as output
        if target.size(1) != output.size(1):
            target = target.unsqueeze(1)  # Adding channel dimension to target if necessary

        # Apply softmax to output to get probabilities
        # output = torch.softmax(output, dim=1)  # Output shape: (NxCxHxW)

        # Flatten the output and target
        output_flat = output.view(-1)
        target_flat = target.view(-1)

        # Convert target to one-hot encoding
        num_classes = output.size(1)
        target_one_hot = torch.zeros_like(output).view(-1, num_classes)  # Shape: (N*H*W, num_classes)
        target_one_hot.scatter_(1, target_flat.unsqueeze(1), 1)  # Fill target_one_hot with correct class indices
        target_one_hot = target_one_hot.view(-1)

        # Calculate intersection and union
        intersection = (output_flat * target_one_hot).sum(dim=0)
        union = (output_flat + target_one_hot).sum(dim=0)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1 - dice_score.mean()  # Take mean of dice scores over all classes