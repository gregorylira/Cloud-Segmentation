import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet18
from torch.nn.functional import interpolate
import lightning as L
from utils import Utils
from fpnHead import FPNHead
from torch.optim.lr_scheduler import ReduceLROnPlateau


class FPN(L.LightningModule):
    def __init__(self, config, pretrained=False, norm_layer=nn.BatchNorm2d):
        super(FPN, self).__init__()
        self.backbone = config.backbone
        self.num_classes = config.num_classes
        self.config = config

        if config.backbone == 'resnet18':
            self.pretrained = resnet18(pretrained=pretrained)
            self.base_forward = self._resnet_base_forward

        in_chs_dict = {"resnet18": 512}  # Update the input channels based on your backbone
        in_chs = in_chs_dict[config.backbone]
        self.head = FPNHead(in_chs, config.num_classes, norm_layer)

        self._initialize_weights()

    def _resnet_base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()
        c1, c2, c3, c4 = self.base_forward(x)
        x = self.head(c1, c2, c3, c4)
       # x = torch.sigmoid(x)  # Apply sigmoid activation for binary classification
        x = interpolate(x, size=(h, w), mode='bilinear', align_corners=False)  # Resize output to input size
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        iou = Utils.iou_metric(y_hat, y)
        dice_coefficient = Utils.dice_coefficient(y_hat, y)
        accuracy = Utils.accuracy(y_hat, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_dice_coefficient', dice_coefficient, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        iou = Utils.iou_metric(y_hat, y)
        dice_coefficient = Utils.dice_coefficient(y_hat, y)
        accuracy = Utils.accuracy(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice_coefficient', dice_coefficient, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        return optimizer