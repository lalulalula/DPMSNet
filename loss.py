import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(size_average=True)
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.bceloss(pred, gt_masks)
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            loss = self.bceloss(pred, gt_masks)
            return loss
        
class AdaptiveSoftIoUFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, focal_alpha=0.25, focal_gamma=2.0):
        """
        alpha: SoftIoU 与 Focal BCE 的权重比例
        focal_alpha, focal_gamma: 控制 Focal Loss 对虚警的抑制强度
        """
        super().__init__()
        self.alpha = alpha
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def soft_iou_loss(self, pred, gt_masks, smooth=1.0):
        intersection = pred * gt_masks
        loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
        loss = 1 - loss.mean()
        return loss

    def focal_bce_loss(self, pred, target):
        """
        改进版：支持分别调整正负样本权重
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1 - pred) * (1 - target)
        
        # 分别处理正负样本的alpha权重
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
    
        return (focal_weight * bce).mean()

    def forward(self, preds, gt_masks):
        if isinstance(preds, (list, tuple)):
            total_loss = 0
            total_iouloss = 0#
            total_focalloss = 0#
            for pred in preds:
                loss_iou = self.soft_iou_loss(pred, gt_masks)
                loss_focal = self.focal_bce_loss(pred, gt_masks)
                scale = (loss_iou.detach() / (loss_focal.detach() + 1e-6)).clamp(0.5, 2)
                # scale = 1.0
                loss = self.alpha * loss_iou + (1 - self.alpha) * scale * loss_focal
                # loss = self.alpha * loss_iou + (1 - self.alpha) * loss_focal
                total_loss += loss
                total_iouloss += loss_iou#
                total_focalloss += loss_focal#
            return total_loss / len(preds)
        else:
            loss_iou = self.soft_iou_loss(preds, gt_masks)
            loss_focal = self.focal_bce_loss(preds, gt_masks)
            scale = (loss_iou.detach() / (loss_focal.detach() + 1e-6)).clamp(0.5, 2)
            # scale = 1.0
            loss = self.alpha * loss_iou + (1 - self.alpha) * scale * loss_focal
            # loss = self.alpha * loss_iou + (1 - self.alpha) * loss_focal
            return loss

class UnscaledSoftIoUFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, focal_alpha=0.25, focal_gamma=2.0):
        """
        alpha: SoftIoU 与 Focal BCE 的权重比例
        focal_alpha, focal_gamma: 控制 Focal Loss 对虚警的抑制强度
        """
        super().__init__()
        self.alpha = alpha
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def soft_iou_loss(self, pred, gt_masks, smooth=1.0):
        intersection = pred * gt_masks
        loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
        loss = 1 - loss.mean()
        return loss

    def focal_bce_loss(self, pred, target):
        """
        改进版：支持分别调整正负样本权重
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1 - pred) * (1 - target)
        
        # 分别处理正负样本的alpha权重
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
    
        return (focal_weight * bce).mean()

    def forward(self, preds, gt_masks):
        if isinstance(preds, (list, tuple)):
            total_loss = 0
            total_iouloss = 0#
            total_focalloss = 0#
            for pred in preds:
                loss_iou = self.soft_iou_loss(pred, gt_masks)
                loss_focal = self.focal_bce_loss(pred, gt_masks)
                # scale = (loss_iou.detach() / (loss_focal.detach() + 1e-6)).clamp(0.5, 2)
                scale = 1.0
                loss = self.alpha * loss_iou + (1 - self.alpha) * scale * loss_focal
                # loss = self.alpha * loss_iou + (1 - self.alpha) * loss_focal
                total_loss += loss
                total_iouloss += loss_iou#
                total_focalloss += loss_focal#
            return total_loss / len(preds),total_iouloss / len(preds),total_focalloss / len(preds)
        else:
            loss_iou = self.soft_iou_loss(preds, gt_masks)
            loss_focal = self.focal_bce_loss(preds, gt_masks)
            # scale = (loss_iou.detach() / (loss_focal.detach() + 1e-6)).clamp(0.5, 2)
            scale = 1.0
            loss = self.alpha * loss_iou + (1 - self.alpha) * scale * loss_focal
            # loss = self.alpha * loss_iou + (1 - self.alpha) * loss_focal
            return loss,loss_iou,loss_focal


class CrossModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_feats, text_feats):
        """
        image_feats: [B, D]
        text_feats: [B, D]
        """
        # L2-normalize
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        # 相似度矩阵：B × B
        logits = image_feats @ text_feats.T / self.temperature

        labels = torch.arange(image_feats.size(0), device=image_feats.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2



class FocalLoss(nn.Module):
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0):

        super(FocalLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def focal_bce_loss(self, pred, target):
        """
        改进版：支持分别调整正负样本权重
        """
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1 - pred) * (1 - target)
        
        # 分别处理正负样本的alpha权重
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        focal_weight = alpha_t * (1 - pt) ** self.focal_gamma
    
        return (focal_weight * bce).mean()

    def forward(self, preds, gt_masks):
        if isinstance(preds, (list, tuple)):
            total_loss = 0
            for pred in preds:
                loss = self.focal_bce_loss(pred, gt_masks)
                total_loss += loss
            return total_loss / len(preds)
        else:
            loss = self.focal_bce_loss(preds, gt_masks)
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge
