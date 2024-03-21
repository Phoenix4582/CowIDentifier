# Torch and Torchvision stuff
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

# Lightning stuff
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule

# Misc stuff
import importlib

class DataModuleTerminal:
    def __init__(self, name, current_fold, folds_file):
        self.name = name
        self.current_fold = current_fold
        self.folds_file = folds_file

    def loadDataset(self, train=True):
        # Which split are we after?
        split = "train" if train else "test"

        # return standard torchvision transformations (randomCrop, resize, normalise)
        # ds_transforms = None if args.model != "TripletTransformer" else get_ds_transforms()
        ds_transforms = transforms.Compose([ transforms.Resize(256),
                                             transforms.RandomCrop(224, pad_if_needed=True),
                                             # transforms.RandomHorizontalFlip(),
                                             # transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

        # Load the selected dataset
        dataset_prefix = getattr(importlib.import_module(f'datasets.{self.name}.{self.name}'), self.name)
        dataset = dataset_prefix(self.current_fold,
                                 self.folds_file,
                                 split=split,
                                 transform=True,
                                 combine=True,
                                 suppress_info=True,
                                 dataset_transforms=ds_transforms)

        return dataset

class LightningCowDataModule(LightningDataModule):
    def __init__(self, name:str, imsize:int, current_fold:int, folds_file:str, batch_size:int, num_workers:int, multi_val_loader:bool = False):
        super(LightningCowDataModule, self).__init__()
        self.name = name
        self.imsize = imsize
        self.current_fold = current_fold
        self.folds_file = folds_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.multi_val_loader = multi_val_loader

        self.dataset_transforms = transforms.Compose([ transforms.Resize(256),
                                             transforms.RandomCrop(self.imsize, pad_if_needed=True),
                                             # transforms.RandomHorizontalFlip(),
                                             # transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

    def retrieve_dataset(self, split):
        dataset_prefix = getattr(importlib.import_module(f'datasets.{self.name}.{self.name}'), self.name)
        dataset = dataset_prefix(self.current_fold,
                                 self.folds_file,
                                 split=split,
                                 transform=True,
                                 combine=True,
                                 suppress_info=True,
                                 dataset_transforms=self.dataset_transforms)

        return dataset

    def setup(self, stage):
        self.train_set = self.retrieve_dataset(split="train")
        self.test_set = self.retrieve_dataset(split="test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

class MultiCamCowsDataModule(LightningDataModule):
    def __init__(self, name:str = "MultiCamCows", batch_size:int = 16, num_workers:int = 16, train_val_split_ratio=1.0, group_by_camera=True):
        super(MultiCamCowsDataModule, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split_ratio = train_val_split_ratio
        self.group_by_camera = group_by_camera

    def retrieve_dataset(self, split):
        dataset_prefix = getattr(importlib.import_module(f'datasets.{self.name}.{self.name}'), self.name)
        dataset = dataset_prefix(split=split, ratio=self.train_val_split_ratio, group_by_camera=self.group_by_camera)

        return dataset

    def setup(self, stage):
        self.train_set = self.retrieve_dataset(split="train")
        self.val_set = self.retrieve_dataset(split="val")
        self.test_set = self.retrieve_dataset(split="test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def predict_dataloader(self):
    #     return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)
