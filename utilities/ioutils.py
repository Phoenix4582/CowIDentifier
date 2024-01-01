# Core libraries
import os
import numpy as np
from PIL import Image

"""
File contains input/output utility functions
"""

# Create a sorted list of all files with a given extension at a given directory
# If full_path is true, it will return the complete path to that file
def allFilesAtDirWithExt(directory, file_extension, full_path=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
    assert os.path.isdir(directory)

    # Gather the files inside
    if full_path:
        files = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]
    else:
        files = [x for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]

    return files

# Create a sorted list of all files with a given extension at a given directory
# If full_path is true, it will return the complete path to that file
# Extend allFilesAtDirWithExt() method with train/val split and camera id filtering
# RETURN BOTH TRAIN AND VAL LIST INSTEAD OF ONE
def splitFilesAtDirWithExt(directory, file_extension, full_path=True, ratio=0.8, group_by_camera=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
    assert os.path.isdir(directory)

    # Make sure ratio is between 0 and 1 (excluding one cause not necessary for spliting)
    assert 0 < ratio < 1

    # Gather the files inside
    if full_path:
        files = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]
    else:
        files = [x for x in sorted(os.listdir(directory)) if x.endswith(file_extension)]

    if not group_by_camera:
        train_files, val_files = files[:int(ratio * len(files))], files[int(ratio * len(files)):]
    else:
        catalogue = {}
        file_camera_ids = retrieveCameraIDFromFiles(files)
        for file, camera_id in zip(files, file_camera_ids):
            if camera_id not in catalogue.keys():
                catalogue[camera_id] = [file]
            else:
                catalogue[camera_id].append(file)

        train_catalogue = {k:v[:int(ratio * len(v))] for k, v in catalogue.items()}
        val_catalogue = {k:v[int(ratio * len(v)):] for k, v in catalogue.items()}

        train_files = [t_file for t_entry in train_catalogue.values() for t_file in t_entry]
        val_files = [v_file for v_entry in val_catalogue.values() for v_file in v_entry]

    assert len(val_files) > 0
    return train_files, val_files

# For a given filepath, return the encoded camera information within the filename
# If no camera information is encoded in "<img_id>_<camera_id> format, return 0"
def retrieveCameraIDFromFiles(files):
    return [filepath[:-4].split("_")[-1] if "_" in filepath else "0" for filepath in files]

# Similarly, create a sorted list of all folders at a given directory
def allFoldersAtDir(directory, full_path=True):
    # Make sure we're looking at a folder
    if not os.path.isdir(directory): print(directory)
    assert os.path.isdir(directory)

    # Find all the folders
    if full_path:
        folders = [os.path.join(directory, x) for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]
    else:
        folders = [x for x in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, x))]

    return folders

# Load an image into memory, pad it to img size with a black background
def loadResizeImage(img_path, size):
    # Load the image
    img = Image.open(img_path)

    # Keep the original image size
    old_size = img.size

    # Compute resizing ratio
    ratio = float(size[0])/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # Actually resize it
    img = img.resize(new_size, Image.ANTIALIAS)

    # Paste into centre of black padded image
    new_img = Image.new("RGB", (size[0], size[1]))
    new_img.paste(img, ((size[0]-new_size[0])//2, (size[1]-new_size[1])//2))

    # Convert to numpy
    new_img = np.array(new_img, dtype=np.uint8)

    return new_img

# Load an image into memory, pad it with a black background WITHOUT resizing
def loadOriginalImage(img_path):
    # Load the image
    img = Image.open(img_path)

    # # Keep the original image size
    # old_size = img.size
    #
    # # Compute resizing ratio
    # ratio = float(size[0])/max(old_size)
    # new_size = tuple([int(x*ratio) for x in old_size])
    #
    # # Actually resize it
    # img = img.resize(new_size, Image.ANTIALIAS)
    #
    # # Paste into centre of black padded image
    # new_img = Image.new("RGB", (size[0], size[1]))
    # new_img.paste(img, ((size[0]-new_size[0])//2, (size[1]-new_size[1])//2))

    # Convert to numpy
    new_img = np.array(img, dtype=np.uint8)

    return new_img
