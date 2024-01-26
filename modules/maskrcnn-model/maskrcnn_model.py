import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import functional as F

class MaskRCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNNModel, self).__init__()
        # Carregar um modelo pré-treinado
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        
        # Modificar a camada de predição da máscara para se adequar ao número de classes desejado
        in_features_mask = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    def forward(self, x, targets=None):
        if self.training and targets is not None:
            loss_dict = self.model(x, targets)
            return loss_dict
        else:
            return self.model(x)
