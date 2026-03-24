# Core libraries
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
# PyTorch
import torch
from torch.utils import data
from torch.autograd import Variable
# Local libraries
from utilities.imageUtils import ImageUtils

# makedirs, KNNAccuracy, inferEmbedding_from_folder, inferEmbedding_from_folder_3, inferEmbeddings

def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def inferEmbedding_from_folder(args, model):
    # Wrap up the dataset in a PyTorch dataset loader
    m = 0
    outputs_embedding = np.zeros((1, args.embedding_size))
    labels_embedding = np.zeros((1))

    i = 1
    count_class = 0
    list = []
    label = []
    list_name = []
    list_name_tmp = []

    for items in os.listdir(args.test_dir):
        count_flag = 0
        for image in os.listdir(os.path.join(args.test_dir, items)):
            count_flag = 1
            img1 = cv2.imread(os.path.join(args.test_dir, items, image))
            list.append(img1)
            if items.isdigit():
                label.append(int(items))
            else:
                label.append(items)
            list_name_tmp.append(os.path.join(items, image))
            if i % 4 ==0:
                for ittt in list_name_tmp:
                    list_name.append(ittt)
                    list_name_tmp = []
                images = ImageUtils.npToTorch([list[0],list[1],list[2],list[3]], return_tensor=True, resize=(224, 224))
                with torch.no_grad():
                    # Put the images on the GPU and express them as PyTorch variables
                    images = Variable(images.cuda())
                    # Get the embeddings of this batch of images
                    outputs = model(images)
                    m += args.batch_size

                    # Express embeddings in numpy form
                    embeddings = outputs.data
                    embeddings = embeddings.detach().cpu().numpy()

                    # Convert labels to readable numpy form
                    labels = np.array(label)
                    l = len(labels)
                    #labels = labels.view(len(labels))
                    #labels = labels.detach().cpu().numpy()

                    # Store testing data on this batch ready to be evaluated
                    outputs_embedding = np.concatenate((outputs_embedding, embeddings), axis=0)
                    labels_embedding = np.concatenate((labels_embedding, labels), axis=0)

                list = []
                label = []
            i += 1
        if count_flag:
            count_class +=1

        # We don't need the autograd engine
    print(f'number of class is {count_class}')
    return np.delete(outputs_embedding, 0, axis = 0), np.delete(labels_embedding, 0), list_name
