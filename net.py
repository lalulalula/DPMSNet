from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        
        # 定义损失计算函数 cal_loss，默认使用 SoftIoULoss()
        self.cal_loss = SoftIoULoss()

        # 选择不同模型
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ResUNet':
            self.model = ResUNet()
            
        
    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss

class DPMSNetWrapper(nn.Module):
    def __init__(self, model_name, mode, loss="AdaptiveSoftIoUFocalLoss"):
        super().__init__()
        self.model_name = model_name
        self.contrastive_loss = CrossModalContrastiveLoss()
        # 定义损失计算函数 cal_loss，默认使用 SoftIoULoss()
        if loss == "SoftIoULoss":
            self.cal_loss = SoftIoULoss()
        elif loss == "AdaptiveSoftIoUFocalLoss":
            self.cal_loss = AdaptiveSoftIoUFocalLoss()
        elif loss == "FocalLoss":
            self.cal_loss = FocalLoss()
        elif loss == "BCELoss":
            self.cal_loss = BCELoss()
        elif loss == "UnscaledSoftIoUFocalLoss":
            self.cal_loss = UnscaledSoftIoUFocalLoss()
        # 选择不同模型
        if model_name == "DPMSNet":
            if mode == 'train':
                self.model = DPMSNet(mode='train')
            else:
                self.model = DPMSNet(mode='test')
            
            
        
    def forward(self, img,txt):
        return self.model(img,txt)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
    