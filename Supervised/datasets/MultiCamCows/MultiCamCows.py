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
from torchvision import transforms

# Local libraries
from utilities.ioutils import *

class MultiCamCows(data.Dataset):
    def __init__(self, split, ratio=1.0, group_by_camera=True):
        # Root directory
        self.__root = "datasets/MultiCamCows"

        # Split parameter for categorising current duty of dataset(i.e. train/test/val/predict)
        self.__split = split

        # train/val split ratio
        self.__split_ratio = ratio

        self.__transforms = transforms.Compose([transforms.ToTensor(),
                                                transforms.Resize((256, 256)),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


        self.__files = {}
        self.__sorted_files = {}

        # The directory containing actual imagery
        self.__train_images_dir = os.path.join(self.__root, "images/train")
        self.__test_images_dir = os.path.join(self.__root, "images/test")

        # Retrieve the number of classes from these
        self.__train_folders = allFoldersAtDir(self.__train_images_dir)
        self.__test_folders = allFoldersAtDir(self.__test_images_dir)
        self.__num_classes = len(self.__train_folders)
        self.__num_test_classes = len(self.__test_folders)

        # Create dictionaries of categories: filepaths
        if self.__split_ratio < 1:
            train_files = {}
            val_files = {}
            for f in self.__train_folders:
                train_list, val_list = splitFilesAtDirWithExt(f, ".jpg", ratio=self.__split_ratio, group_by_camera=group_by_camera)
                train_files[os.path.basename(f)] = train_list
                val_files[os.path.basename(f)] = val_list
            # train_files = {os.path.basename(f):splitFilesAtDirWithExt(f, ".jpg", ratio=self.__split_ratio, group_by_camera=group_by_camera)[0] for f in self.__train_folders}
            # val_files = {os.path.basename(f):splitFilesAtDirWithExt(f, ".jpg", ratio=self.__split_ratio, group_by_camera=group_by_camera)[1] for f in self.__train_folders}
        else:
            train_files = {os.path.basename(f):allFilesAtDirWithExt(f, ".jpg") for f in self.__train_folders}
        test_files = {os.path.basename(f):allFilesAtDirWithExt(f, ".jpg") for f in self.__test_folders}

        self.__sorted_files['train'] = {k:v for (k,v) in train_files.items()}
        self.__sorted_files['test'] = {k:v for (k,v) in test_files.items()}

        train_list = [v for k,v in train_files.items()]
        test_list = [v for k,v in test_files.items()]
        self.__files['train'] = [item for sublist in train_list for item in sublist]
        self.__files['test'] = [item for sublist in test_list for item in sublist]

        if self.__split_ratio < 1:
            val_list = [v for k,v in val_files.items()]
            self.__sorted_files['val'] = {k:v for (k,v) in val_files.items()}
            self.__files['val'] = [item for sublist in val_list for item in sublist]
        else:
            self.__sorted_files['val'] = {k:v for (k,v) in test_files.items()}
            self.__files['val'] = [item for sublist in test_list for item in sublist]

    def __len__(self):
        return len(self.__files[self.__split])

    def __getitem__(self, index):
        # Get and load the anchor image
        img_path = self.__files[self.__split][index]

        # Retrieve the class/label this index refers to
        current_category = self.__retrieveCategoryForFilepath(img_path)
        current_camera = self.__retrieveCameraIDFromImg(img_path)

        # img_anchor = loadResizeImage(img_path, (256, 256))
        img_anchor = Image.open(img_path)
        # img_anchor = np.transpose(img_anchor, (2, 0, 1)) # H * W * C -> C * H * W
        img_anchor = self.__transforms(img_anchor)
        label = np.array([int(current_category)])
        camera = np.array([int(current_camera)])

        return img_anchor, label, camera

    # For a given filepath, return the category which contains this filepath
    def __retrieveCategoryForFilepath(self, filepath):
        # Iterate over each category
        for category, filepaths in self.__sorted_files[self.__split].items():
            if filepath in filepaths: return category

    # For a given filepath, return the encoded camera information within the filename
    # If no camera information is encoded in "<img_id>_<camera_id> format, return 0"
    def __retrieveCameraIDFromImg(self, filepath):
        if "_" not in filepath:
            return 0
        return filepath[:-4].split("_")[-1]
