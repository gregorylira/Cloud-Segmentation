from pathlib import Path
from unet_model import UNET
import warnings
from configs import Configs
from fpn_model import FPN
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import os
from PIL import Image
import cv2

def stack_bands(r, g, b, nir = None):
        raw_rgb = np.stack([
            np.array(Image.open(r)),
            np.array(Image.open(g)),
            np.array(Image.open(b)),
        ], axis=2)
    
        if nir:
            nir = np.expand_dims(
                np.array(Image.open(nir)), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
        
        return raw_rgb
    

def img2tensor(img):
    img = img.transpose((2, 0, 1))
    img = img / np.iinfo(img.dtype).max
    img = torch.tensor(img, dtype=torch.float32)
    return img.unsqueeze(0)
    

def predb_to_mask_predict(predb):
    pred_probabilities = torch.nn.functional.softmax(predb, dim=1)
    _, predicted_class = torch.max(pred_probabilities, dim=1)
    return predicted_class.cpu().numpy()

def predb_to_mask_threshold(predb, threshold=0.5):
    # Assuming predb values are in the range [-1, 1]
    mask = (predb[:, 1, :, :] > threshold).cpu().numpy().astype(np.uint8)
    return mask

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--images_path', type=str, help='Image path')

    args = parser.parse_args()

    base_path = Path(args.images_path)

    pre_process = transforms.Compose([
        transforms.Resize((384, 384)),
    ])

    r_path = base_path/'red/red_image.TIFF'
    g_path = base_path/'green/green_image.TIFF'
    b_path = base_path/'blue/blue_image.TIFF'

    raw_rgb = stack_bands(r_path, g_path, b_path)
    raw_rgb = img2tensor(raw_rgb)
    raw_rgb = pre_process(raw_rgb)
 
    # freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    config = Configs()
    torch.manual_seed(config.seed)

    if config.model == 'fpn':
        model = FPN.load_from_checkpoint(config.model_checkpoint, in_channels=3, out_channels=2, config=config)
    elif config.model == 'unet':
        model = UNET.load_from_checkpoint(config.model_checkpoint, in_channels=3, out_channels=2, config=config)
    elif config.model == 'deeplabv3':
        pass

    model.eval()
    if config.device == 'gpu':
        model.to('cuda')
        raw_rgb = raw_rgb.to('cuda')
    
    with torch.no_grad():
        pred = model(raw_rgb)
    
    
    input_image = raw_rgb[0, 0:3].cpu().numpy().transpose((1, 2, 0))
    # pred = pred.unsqueeze(0)
    pred = predb_to_mask_predict(pred)
    # pred = predb_to_mask_threshold(pred)
    print(pred.shape)
    fig, ax = plt.subplots(1, 2, figsize=(10, 9))
    ax[0].imshow(input_image)
    ax[1].imshow(pred[0], cmap='gray')
    plt.show()  

    #  if image already exists, create a new name for the image
    if os.path.exists(f'{base_path}/imagem_plot.png'):
        i = 1
        while os.path.exists(f'{base_path}/imagem_plot_{i}.png'):
            i += 1
        fig.savefig(f'{base_path}/imagem_plot_{i}.png')
    else:
        fig.savefig(f'{base_path}/imagem_plot.png')



    mask_expanded = np.zeros_like(input_image)
    mask_expanded[pred[0] > 0] = [255, 0, 0]  

    combined_image = np.maximum(input_image, mask_expanded)

    fig, ax = plt.subplots(figsize=(10, 9))
    plt.imshow(combined_image)
    plt.show()

    if os.path.exists(f'{base_path}/image_show_union.png'):
        i = 1
        while os.path.exists(f'{base_path}/image_show_union_{i}.png'):
            i += 1
        fig.savefig(f'{base_path}/image_show_union_{i}.png')
    else:
        fig.savefig(f'{base_path}/image_show_union.png')
