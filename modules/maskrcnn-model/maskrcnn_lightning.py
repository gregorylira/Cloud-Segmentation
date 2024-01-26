import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader
import lightning as L
from torchvision.transforms import functional as F


class MaskRCNNLightning(L.LightningModule):
    def __init__(self, model, optimizer, scheduler=None):
        super(MaskRCNNLightning, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x, targets=None):
        return self.model(x, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', total_loss)
        return total_loss

    def configure_optimizers(self):
        if self.scheduler is not None:
            return [self.optimizer], [self.scheduler]
        else:
            return self.optimizer