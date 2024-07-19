# Core libraries
import os
import sys
import cv2
import json
import random
import numpy as np
from PIL import Image

# PyTorch
import torch
from torch.utils import data
import torchvision

# def collate_fn(batch):
#     images, dates, class_ids = zip(*batch)
#     unique_dates = set(dates)
#     grouped_data = {date: {'images':[], 'classes':[]} for date in unique_dates}
#
#     for image, date, id in zip(images, dates, class_ids):
#         grouped_data[date]['images'].append(image)
#         grouped_data[date]['classes'].append(id)
#
#     batch_imgs = []
#     batch_classes = []
#
#     for date, data in grouped_data.items():
#         batch_imgs.extend(data['images'])
#         batch_classes.extend(data['classes'])
#
#     return batch_imgs, batch_classes


class MultiCamDailyCows2023(data.Dataset):
    def __init__(self, root, date_pin, transform=None, pseudo=True, pseudo_thr=50):
        self.camera_root_id = root

        # fused folder as a separate folder named camera_0
        pseudo_root = f'datasets/MultiCamDailyCows2023/Pseudo/camera_{self.camera_root_id}'
        global_root = f'datasets/MultiCamDailyCows2023/Global/camera_{self.camera_root_id}'
        self.selected_root = pseudo_root if pseudo else global_root
        self.transform = transform
        # self.file_list = self._build_file_list()
        self.dates_subfolders = os.listdir(self.selected_root)
        self.file_dict = self._build_file_dict(date_pin, pseudo, pseudo_thr)
        self.date_pin = date_pin
        self.date_classes = self._get_date_class()
        # self.print_item_list()
        # self.train_file_list = self._build_file_list(specific_date=)
        # self.val_file_list = self._build_file_list(specific_date=)
        # self.test_file_list = self._build_file_list(specific_date=)

    def _build_file_dict(self, date_pin, pseudo=True, pseudo_thr=50):
        assert isinstance(date_pin, list), "Format Error: date_pin variable must be a list of dates, i.e.[2023Aug14, 2023Aug15]"
        file_dict = []
        # type_list = []
        # types = ['train', 'val', 'test']
        # for type, pin in zip(types, date_pin):
        #     type_list.extend([type for _ in range(pin)])

        # date_dict = {date: type for date, type in zip(self.dates_subfolders, type_list)}
        # for date, type in date_dict.items():
        #     print(date, type)

        for date in date_pin:
            date_folder = os.path.join(self.selected_root, date)
            for id in os.listdir(date_folder):
                file_folder = os.path.join(date_folder, id)
                subfolder_imgs = [item for item in os.listdir(file_folder) if item.endswith('.jpg')]
                if pseudo and pseudo_thr > 0:
                    entry_imgs = [random.choice(subfolder_imgs) for _ in range(pseudo_thr)]
                else:
                    entry_imgs = subfolder_imgs
                for file in entry_imgs:
                    file_dict.append(os.path.join(self.selected_root, date, id, file))

        return file_dict

    def _get_date_class(self):
        """
        Fetch a dictionary of (date: item_index) from dataset,
        Required for Custom Sampler
        """
        # type_list = ['train', 'val', 'test']
        # data_type_index = type_list.index(self.__type)
        data_source = self.fetch_data()
        unique_dates = set([img_path.split("/")[-3:][0] for img_path in data_source])
        date_class = {date: [] for date in unique_dates}
        for i, data in enumerate(data_source):
            date_folder = data.split("/")[-3:][0]
            date_class[date_folder].append(i)

        return date_class

    def fetch_date_class(self):
        return list(self.date_classes.values())

    def fetch_data(self):
        # assert type in ['train', 'val', 'test']
        return self.file_dict

    def _build_file_list(self):
        file_list = []
        for date in os.listdir(self.selected_root):
            date_folder = os.path.join(self.selected_root, date)
            for id in os.listdir(date_folder):
                file_folder = os.path.join(date_folder, id)
                for file in os.listdir(file_folder):
                    if file.endswith('jpg'):
                        file_dict[date_dict[date]].append(os.path.join(self.selected_root, date, id, file))

        return file_list

    def __len__(self):
        return len(self.file_dict)

    def __getitem__(self, index):
        img_path = self.file_dict[index]
        image = Image.open(img_path).convert('RGB')

        # Extract date and class id from path
        date, class_id, _ = img_path.split("/")[-3:]

        class_id = int(class_id)
        if self.transform:
            image = self.transform(image)

        return image, date, class_id

    def print_item_list(self):
        for key, query in self.file_dict.items():
            print(f"{key}: {query}")

    def get_labels(self):
        # self.print_item_list()
        result = []
        for img_path in self.file_dict:
            _, class_id, _ = img_path.split("/")[-3:]
            class_id = int(class_id)
            result.append(class_id)
        # assert len(result) != 0
        return result
