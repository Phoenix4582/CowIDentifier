# Torch and Torchvision stuff
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Sampler, BatchSampler

# Lightning stuff
import lightning.pytorch as pl
from lightning.pytorch import LightningDataModule

# Misc stuff
import importlib
from torchsampler import ImbalancedDatasetSampler
import numpy as np
import random

class SameDateSampler(BatchSampler):
    """Samples data of same date from data_source.

    Args:
        data_source (list): contains tuples of (imgs, dates, ids).
        batch_size (int): batch size.
    Return
        yields the (imgs, dates, ids)

    """
    def __init__(self, data_source, batch_size, drop_last:bool = False, random_date_per_epoch:bool=True, shuffle:bool=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random_entry = random_date_per_epoch
        self.classes = self.data_source.fetch_date_class()
        self.chosen_class = random.sample(self.classes, 1)[0] if self.random_entry else None
        self.seed = int(torch.empty((), dtype=torch.int64).random_().item())
        random.seed(self.seed)
        self.shuffle = shuffle

    def __len__(self):
        if self.random_entry:
            if self.drop_last:
                return int(len(self.chosen_class) // self.batch_size)
            return int((len(self.chosen_class) + self.batch_size - 1) // self.batch_size)
        else:
            if self.drop_last:
                return int(sum([len(entry) // self.batch_size for entry in self.classes()]))
            return int(sum([(len(entry) + self.batch_size - 1) // self.batch_size for entry in self.classes()]))

    # def get_data_source_class_list(self):
    #     assert(hasattr(self.data_source, 'fetch_date_class'))
    #     assert(callable(self.data_source.fetch_date_class))
    #     return self.data_source.fetch_date_class()

    def __iter__(self):
        if self.random_entry:
            chosen_class = self.chosen_class
            if self.shuffle:
                random.shuffle(chosen_class)
            for i in range(0, len(self.chosen_class), self.batch_size):
                target = chosen_class[i:i+self.batch_size]
                if len(target) == self.batch_size or not self.drop_last:
                    yield target
        else:
            for class_list in self.classes:
                if self.shuffle:
                    random.shuffle(class_list)
                for i in range(0, len(class_list), self.batch_size):
                    target = class_list[i:i+self.batch_size]
                    if len(target) == self.batch_size or not self.drop_last:
                        yield target


def custom_collate_fn(batch):
    images, dates, class_ids = zip(*batch)

    # return torch.utils.data.default_collate(images), torch.utils.data.default_collate(class_ids), torch.utils.data.default_collate(dates)
    return torch.utils.data.default_collate(images), torch.utils.data.default_collate(class_ids)
    # unique_dates = set(dates)
    # grouped_data = {date: {'images':[], 'classes':[]} for date in unique_dates}
    #
    # for image, date, id in zip(images, dates, class_ids):
    #     grouped_data[date]['images'].append(image)
    #     grouped_data[date]['classes'].append(id)
    #
    # batch_imgs = []
    # batch_classes = []
    # batch_date_origins = []
    #
    # for date, data in grouped_data.items():
    #     for img, classes in zip(data['images'], data['classes']):
    #         batch_imgs.append(img)
    #         batch_classes.append(classes)
    #         batch_date_origins.append(date)
    #     # batch_imgs.extend(data['images'])
    #     # batch_classes.extend(data['classes'])
    # return torch.utils.data.default_collate(batch_imgs), torch.utils.data.default_collate(batch_classes), torch.utils.data.default_collate(batch_date_origins)

class MultiCamDailyCowsDataModule(LightningDataModule):
    def __init__(self, name:str = "MultiCamDailyCows2023",
                       root:int = 1,
                       train_pin:list = ['2023Aug14', '2023Aug15', '2023Aug16', '2023Aug17', '2023Aug18'],
                       val_pin:list = ['2023Aug19'],
                       test_pin:list = ['2023Aug20'],
                       batch_size:int = 16,
                       num_workers:int = 16,
                       pseudo_thr:int = 50,
                       balanced_sample:bool = True):
        super(MultiCamDailyCowsDataModule, self).__init__()
        self.name = name
        self.root = root
        self.date_pin = {'train': train_pin, 'val': val_pin, 'test': test_pin}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pseudo_thr = pseudo_thr
        self.__apply_balanced_sample = balanced_sample
        self.custom_collate_fn = custom_collate_fn
        self.transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((256, 256)),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def retrieve_dataset(self, type, pseudo=True):
        dataset_prefix = getattr(importlib.import_module(f'datasets.{self.name}.{self.name}'), self.name)
        return dataset_prefix(root=self.root, date_pin=self.date_pin[type], transform=self.transform, pseudo=pseudo, pseudo_thr=self.pseudo_thr)

    def setup(self, stage):
        self.train_set = self.retrieve_dataset(type='train')
        self.val_set = self.retrieve_dataset(type='val')

        self.test_set_train_data = self.retrieve_dataset(type='train', pseudo=False)
        self.test_set_test_data = self.retrieve_dataset(type='test', pseudo=False)

    def train_dataloader(self):
        # train_sampler = ImbalancedDatasetSampler(self.train_set) if self.__apply_balanced_sample else None
        # apply_shuffle = False if self.__apply_balanced_sample else True
        # return DataLoader(self.train_set,
        #                   sampler=train_sampler,
        #                   batch_size=self.batch_size,
        #                   shuffle=apply_shuffle,
        #                   num_workers=self.num_workers,
        #                   collate_fn=self.custom_collate_fn)
        same_date_sampler = SameDateSampler(self.train_set, batch_size=self.batch_size, drop_last=False, random_date_per_epoch=True, shuffle=True)
        return DataLoader(self.train_set,
                          batch_sampler=same_date_sampler,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        val_sampler = ImbalancedDatasetSampler(self.val_set) if self.__apply_balanced_sample else None
        return DataLoader(self.val_set,
                          sampler=val_sampler,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        sampler = ImbalancedDatasetSampler(self.test_set) if self.__apply_balanced_sample else None
        test_mode_train_loader = DataLoader(self.test_set_train_data,
                          sampler=sampler,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate_fn)
        test_mode_test_loader = DataLoader(self.test_set_test_data,
                          sampler=sampler,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.custom_collate_fn)
        return [test_mode_train_loader, test_mode_test_loader]
