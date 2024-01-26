import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import lightning as L
from datasetclass_38 import CloudDataset
from transforms_class import DataAugmentations
import random


class CloudDataModule(L.LightningDataModule):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, transforms, batch_size: int = 32, num_workers: int = 4, seed: int = 42):
        super().__init__()
        self.r_dir = r_dir
        self.g_dir = g_dir
        self.b_dir = b_dir
        self.nir_dir = nir_dir
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transforms = transforms

    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        # This is called on every process in DDP
        dataset = CloudDataset(
            self.r_dir, self.g_dir, self.b_dir, self.nir_dir, self.gt_dir, transform=self.transforms)

        # Split your dataset into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))
        print(len(train_dataset), len(val_dataset))

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, persistent_workers=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, persistent_workers=False, num_workers=self.num_workers)