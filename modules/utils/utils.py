import torch
import numpy as np
from PIL import Image

class Utils():
    def __init__(self):
        self.name = 'utils'
        
    def iou_metric(predb, yb, threshold=0.5):
        pred_mask = (predb.argmax(dim=1) > threshold).float()
        true_mask = (yb > 0.5).float()

        intersection = torch.sum(pred_mask * true_mask)
        union = torch.sum((pred_mask + true_mask) > 0)

        iou = intersection / union

        return iou


    def dice_coefficient(predb, yb, threshold=0.5):
        pred_mask = (predb.argmax(dim=1) > threshold).float()
        true_mask = (yb > 0.5).float()

        intersection = torch.sum(pred_mask * true_mask)
        dice_coeff = (2 * intersection) / (torch.sum(pred_mask) + torch.sum(true_mask))

        return dice_coeff

    def accuracy(predb, yb, threshold=0.5):
        pred_mask = (predb.argmax(dim=1) > threshold).float()
        true_mask = (yb > 0.5).float()

        correct_pixels = torch.sum(pred_mask == true_mask)
        total_pixels = pred_mask.numel()

        accuracy = correct_pixels / total_pixels

        return accuracy