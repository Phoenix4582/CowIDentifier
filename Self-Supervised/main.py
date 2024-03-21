# PyTorch stuff
import torch
from torch import optim
from torch.utils import data
import lightning.pytorch as pl
from lightning.pytorch import LightningModule
from lightning.pytorch import LightningDataModule
from lightning.pytorch.cli import LightningCLI

# Import Lightning models  with loss_fn, optimisers, and other miscellaneuous utilities
# from lightning_model import LightningIDModel
from lightning_multicam_model import MultiCamModel

# Import Datasets
# from lightning_data_terminal import DataModuleTerminal
# from lightning_data_terminal import LightningCowDataModule
from lightning_data_terminal import MultiCamDailyCowsDataModule

if __name__ == '__main__':
    # module = MultiCamDailyCowsDataModule()
    # module.setup(stage='train')
    # loader = module.train_dataloader()
    #
    # for data_batch in iter(loader):
    #     print(data_batch[2])
    cli = LightningCLI(MultiCamModel, MultiCamDailyCowsDataModule, parser_kwargs={"parser_mode": "omegaconf"})
