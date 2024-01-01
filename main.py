# PyTorch stuff
import torch
from torch import optim
from torch.utils import data
import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI

# Import Lightning models  with loss_fn, optimisers, and other miscellaneuous utilities
from lightning_model import LightningIDModel
from lightning_multicam_model import MultiCamModel

# Import Datasets
# from lightning_data_terminal import DataModuleTerminal
# from lightning_data_terminal import LightningCowDataModule
from lightning_data_terminal import MultiCamCowsDataModule

if __name__ == '__main__':
    cli = LightningCLI(MultiCamModel, MultiCamCowsDataModule, parser_kwargs={"parser_mode": "omegaconf"})
