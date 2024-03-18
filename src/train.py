from pathlib import Path
from datamodule_38 import CloudDataModule
from unet_model import UNET
import lightning as L
import warnings
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import EarlyStopping
from configs import Configs
from fpn_model import FPN
from maskrcnn_model import MaskRCNNModel
from maskrcnn_lightning import MaskRCNNLightning
import torch
import wandb



if __name__ == "__main__":
    # freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    config = Configs()
    torch.manual_seed(config.seed)


    earlystop_callback = EarlyStopping(
        monitor='losses/val_loss',
        min_delta=0.00,
        patience=config.patience,
        verbose=True,
        mode='min'
    )
    
    if config.data['logger'] == "wandb":
        logger = WandbLogger(name=f"{config.model}_{config.lr}_{config.path_info}", project="38-Cloud", log_model='all', save_dir=config.save_dir_model)
        logger.log_hyperparams(config.data)


    else:
        logger = TensorBoardLogger("logs", name="my_model", save_dir=config.save_dir_model)

    base_path = Path('D:/TCC/selection_cloud/38-cloud/38-Cloud_training')
    transform = True if config.data['transforms'] == 'True' else False
    # transform = False
    data = CloudDataModule(base_path/'train_red',
                        base_path/'train_green',
                        base_path/'train_blue',
                        base_path/'train_nir',
                        base_path/'train_gt',
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        seed = config.seed,
                        transforms=transform
                        )
    

    if config.model == 'fpn':
        model = FPN(config=config, pretrained=False)
    elif config.model == 'unet':
        model = UNET(in_channels=3, out_channels=2, config=config)
    elif config.model == 'deeplabv3':
        pass
    
    if config.device == 'gpu':
        trainer = L.Trainer(logger=logger, accelerator="gpu", devices=1, max_epochs=config.num_epochs, callbacks=[earlystop_callback])
    else:
        trainer = L.Trainer(logger=logger, max_epochs=config.num_epochs, callbacks=[earlystop_callback])
    
    # strategy="ddp"
    trainer.fit(model, data)