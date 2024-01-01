# PyTorch stuff
import torch
from torch.autograd import Variable
from torch import optim
from torch.utils import data

# PyTorch Lightning
from lightning.pytorch import LightningModule

import os
import shutil
import numpy as np
import openTSNE
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
# Misc stuff
import importlib

# Import our own classes
from utilities.loss import *
from utilities.mining_utils import *
# from models.TripletResnet import TripletResnet50
# from models.TripletResnetSoftmax import TripletResnet50Softmax
# from models.TripletTransformer import TripletTransformer
# from models.TripletTransformerSoftmax import TripletTransformerSoftmax
from models.embeddings import resnet50
from models.embeddings_transformer import ViT

from utilities.utils_misc import KNNAccuracy, fetch_npz_data
from utilities.utils_tsne import prepare_vis, prepare_kmeans, prepare_gmm


class LightningIDModel(LightningModule):
    def __init__(self, model:str, num_classes:int, triplet_selector:str, loss_fn:str, save_path:str, ckpt_base:str, vit_backbone='vit_b_16', triplet_margin=0.5, include_softmax=True, embedding_size=128):
        # lr=0.001, weight_decay=1e-4
        super(LightningIDModel, self).__init__()
        self.include_softmax = "Softmax" in loss_fn
        self.reciprocal = "Reciprocal" in loss_fn

        self.save_path = save_path
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        os.makedirs(self.save_path)

        self.include_softmax = include_softmax
        self.embedding_size = embedding_size

        # Four variables for valid/test steps
        self.ckpt_base = ckpt_base
        self.num_classes = num_classes
        self.model = self.fetchModel(model, num_classes, vit_backbone)
        # self.test_model = self.fetchTestModel(model, num_classes, vit_backbone)
        # self.optimiser = self.fetchOptimiser(self.model, lr, weight_decay)
        self.triplet_selector = self.fetchTripletSelector(triplet_selector, triplet_margin)
        self.loss_fn = self.fetchLossFunction(loss_fn, self.triplet_selector, triplet_margin)

        self.tsne = self.init_tsne()
        self.umap = self.init_umap()
        self.kmeans = self.init_kmeans(num_classes)
        self.gmm = self.init_gmm(num_classes)

        self.save_hyperparameters()

    def fetchModel(self, name, num_classes, vit_backbone):
        model_prefix = getattr(importlib.import_module(f'models.{name}'), name)
        if "Transformer" in name:
            model = model_prefix(backbone=vit_backbone, pretrained=True, num_classes=num_classes)
        elif "Resnet" in name:
            model = model_prefix(pretrained=True, num_classes=num_classes)
        else:
            print(f"Model choice: \"{name}\" not recognised, exiting.")
            sys.exit(1)
            
        return model

    def fetchTestModel(self, name, num_classes, vit_backbone):
        ckpt_path = [os.path.join(self.ckpt_base, ckpt) for ckpt in os.listdir(self.ckpt_base) if ckpt.endswith(".ckpt")]
        ckpt_path = ckpt_path[-1]
        if "Resnet" in name:
            test_model = resnet50(pretrained=True, num_classes=self.num_classes, ckpt_path=ckpt_path, embedding_size=self.embedding_size, lightning_mode=True)
        elif "Transformer" in name:
            test_model = ViT(backbone=self.vit_backbone, pretrained=True, num_classes=self.num_classes, ckpt_path=ckpt_path, embedding_size=self.embedding_size, lightning_mode=True)
        else:
            print(f"Test Model choice: \"{name}\" not recognised, exiting.")
            sys.exit(1)

        return test_model

    def fetchTripletSelector(self, name, triplet_margin):
        # Setup the triplet selection method
        if name == "HardestNegative":
            triplet_selector = HardestNegativeTripletSelector(margin=triplet_margin)
        elif name == "RandomNegative":
            triplet_selector = RandomNegativeTripletSelector(margin=triplet_margin)
        elif name == "SemihardNegative":
            triplet_selector = SemihardNegativeTripletSelector(margin=triplet_margin)
        elif name == "AllTriplets":
            triplet_selector = AllTripletSelector()
        else:
            print(f"Triplet selection choice not recognised, exiting.")
            sys.exit(1)

        return triplet_selector

    def fetchLossFunction(self, loss_name, triplet_selector, triplet_margin):
        # Setup the selected loss function
        if loss_name == "TripletLoss":
            loss_fn = TripletLoss(margin=triplet_margin)
        elif loss_name == "TripletSoftmaxLoss":
            loss_fn = TripletSoftmaxLoss(margin=triplet_margin)
        elif loss_name == "OnlineTripletLoss":
            loss_fn = OnlineTripletLoss(triplet_selector, margin=triplet_margin)
        elif loss_name == "OnlineTripletSoftmaxLoss":
            loss_fn = OnlineTripletSoftmaxLoss(triplet_selector, margin=triplet_margin)
        elif loss_name == "OnlineReciprocalTripletLoss":
            loss_fn = OnlineReciprocalTripletLoss(triplet_selector)
        elif loss_name == "OnlineReciprocalSoftmaxLoss":
            loss_fn = OnlineReciprocalSoftmaxLoss(triplet_selector)
        else:
            print(f"Loss function choice not recognised, exiting.")
            sys.exit(1)

        return loss_fn

    def init_tsne(self, n_components=2, perplexity=30):
        # Initialise a TSNE model with given parameters from OpenTSNE library.
        tsne = openTSNE.TSNE(n_components=n_components,
                             perplexity=perplexity,
                             metric="cosine",
                             n_jobs=8,
                             random_state=42,
                             verbose=False,)

        return tsne

    def init_kmeans(self, num_classes):
        grouper = KMeans(n_clusters=num_classes, n_init=10)
        return grouper

    def init_gmm(self, num_classes):
        gmm = GaussianMixture(n_components=num_classes, covariance_type="full")
        return gmm

    # def fetchOptimiser(self, model, learning_rate, weight_decay):
    #     # Create our optimiser, if using reciprocal triplet loss, don't have a momentum component
    #     if self.reciprocal:
    #         optimiser = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #         # optimiser = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    #     else:
    #         optimiser = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    #
    #     return optimiser

    def run_model(self, batch, batch_idx, mode="train"):
        base_loss = "loss" if mode == "train" else "val_loss"
        if mode == "train":
            opt = self.optimizers()
        for images, images_pos, images_neg, labels, labels_neg, camera_id, camera_neg in (batch,):
            images = Variable(images)
            images_pos = Variable(images_pos)
            images_neg = Variable(images_neg)
            if mode == "train":
                opt.zero_grad()
            # Get the embeddings/predictions for each
            if self.include_softmax:
                embed_anch, embed_pos, embed_neg, preds = self.model(images, images_pos, images_neg)
            else:
                embed_anch, embed_pos, embed_neg = self.model(images, images_pos, images_neg)

            # Calculate the loss on this minibatch
            if self.include_softmax:
                loss, triplet_loss, loss_softmax = self.loss_fn(embed_anch, embed_pos, embed_neg, preds, labels, labels_neg)
                # loss_dict = {base_loss:loss.item(), f"{base_loss}_triplet": triplet_loss.item(), f"{base_loss}_softmax": loss_softmax.item()}
                loss_dict = {base_loss:loss, f"{base_loss}_triplet": triplet_loss, f"{base_loss}_softmax": loss_softmax}
                # self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
            else:
                loss = self.loss_fn(embed_anch, embed_pos, embed_neg, labels)
                # loss_dict = {base_loss:loss.item()}
                loss_dict = {base_loss:loss}

        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        if mode == "train":
            self.manual_backward(loss)
            opt.step()
        # return loss

    def training_step(self, batch, batch_idx):
        # print(next(self.models.parameters()).device)
        self.run_model(batch, batch_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx != 0:
            # print("Calculating validation loss......")
            self.run_model(batch, batch_idx, mode="val")
        if self.current_epoch == 0:
            return
        # print("Running KNNAccuracy......")
        # Construct the save path
        test_model = self.fetchTestModel(self.model_name, self.num_classes, self.vit_backbone)
        test_model.cuda()
        name = "train" if dataloader_idx == 0 else "test"
        npz_save_path = os.path.join(self.save_path, f"{name}_embeddings.npz")
        if os.path.exists(npz_save_path):
            data = np.load(npz_save_path)
            outputs_embedding = data['embeddings']
            labels_embedding = data['labels']
            camera_embedding = data['camera']
        else:
            # Embeddings/labels to be stored on the testing set
            outputs_embedding = None
            labels_embedding = None
            camera_embedding = None

        # Iterate through the testing portion of the dataset and get
        for images, _, _, labels, _, camera_id, _ in (batch,):
            # Put the images on the GPU and express them as PyTorch variables
            images = Variable(images.cuda())

            # Get the embeddings of this batch of images
            outputs = test_model(images)

            # Express embeddings in numpy form
            embeddings = outputs.data
            embeddings = embeddings.cpu().numpy()
            # print(f"Embedding shape:{embeddings.shape}")

            # Convert labels to readable numpy form
            labels = labels.view(len(labels))
            labels = labels.cpu().numpy()
            # print(f"Labels shape:{labels.shape}")

            # Convert camera ID to readable numpy form
            camera_id = camera_id.view(len(camera_id))
            camera_id = camera_id.cpu().numpy()

            # Store testing data on this batch ready to be evaluated
            outputs_embedding = embeddings if outputs_embedding is None else np.concatenate((outputs_embedding, embeddings), axis=0)
            labels_embedding = labels if labels_embedding is None else np.concatenate((labels_embedding, labels), axis=0)
            camera_embedding = camera_id if camera_embedding is None else np.concatenate((camera_embedding, camera_id), axis=0)

        np.savez(npz_save_path, embeddings=outputs_embedding, labels=labels_embedding, camera=camera_embedding)

    def on_validation_epoch_end(self):
        if self.current_epoch == 0:
            return
        train_embeddings, train_labels = fetch_npz_data(os.path.join(self.save_path, "train_embeddings.npz"))
        test_embeddings, test_labels = fetch_npz_data(os.path.join(self.save_path, "test_embeddings.npz"))
        knn_accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)
        # print(f"KNNAccuracy: {knn_accuracy} %")
        self.log_dict({"val_acc": knn_accuracy})
        title_str = f"Epoch-{self.current_epoch}; Accuracy-{round(knn_accuracy, 3)}%"
        prepare_vis(self.current_epoch, self.num_classes, self.save_path, self.tsne, title_str)
        prepare_kmeans(self.current_epoch, self.num_classes, self.save_path, self.kmeans, title_str)
        prepare_gmm(self.current_epoch, self.num_classes, self.save_path, self.gmm, title_str)
        os.remove(os.path.join(self.save_path, "train_embeddings.npz"))
        os.remove(os.path.join(self.save_path, "test_embeddings.npz"))

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # if os.path.exists(self.ckpt_base):
        test_model = self.fetchTestModel(self.name, self.num_classes, self.vit_backbone)
        test_model.cuda()
        # Construct the save path
        name = "train" if dataloader_idx == 0 else "test"
        npz_save_path = os.path.join(self.save_path, f"{name}_embeddings.npz")
        if os.path.exists(npz_save_path):
            data = np.load(npz_save_path)
            outputs_embedding = data['embeddings']
            labels_embedding = data['labels']
            camera_embedding = data['camera']
        else:
            # Embeddings/labels to be stored on the testing set
            outputs_embedding = None
            labels_embedding = None
            camera_embedding = None

        # Iterate through the testing portion of the dataset and get
        for images, _, _, labels, _, camera_id, _ in (batch,):
            # Put the images on the GPU and express them as PyTorch variables
            images = Variable(images)

            # Get the embeddings of this batch of images
            outputs = test_model(images)

            # Express embeddings in numpy form
            embeddings = outputs.data
            embeddings = embeddings.cpu().numpy()
            # print(f"Embedding shape:{embeddings.shape}")

            # Convert labels to readable numpy form
            labels = labels.view(len(labels))
            labels = labels.cpu().numpy()
            # print(f"Labels shape:{labels.shape}")

            # Convert camera ID to readable numpy form
            camera_id = camera_id.view(len(camera_id))
            camera_id = camera_id.cpu().numpy()

            # Store testing data on this batch ready to be evaluated
            outputs_embedding = embeddings if outputs_embedding is None else np.concatenate((outputs_embedding, embeddings), axis=0)
            labels_embedding = labels if labels_embedding is None else np.concatenate((labels_embedding, labels), axis=0)
            camera_embedding = camera_id if camera_embedding is None else np.concatenate((camera_embedding, camera_id), axis=0)

        np.savez(npz_save_path, embeddings=outputs_embedding, labels=labels_embedding, camera=camera_embedding)

    def on_test_epoch_end(self):
        if self.current_epoch == 0:
            return
        train_embeddings, train_labels = fetch_npz_data(os.path.join(self.save_path, "train_embeddings.npz"))
        test_embeddings, test_labels = fetch_npz_data(os.path.join(self.save_path, "test_embeddings.npz"))
        knn_accuracy = KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels)
        print(f"KNNAccuracy: {knn_accuracy} %")
        self.log_dict({"val_acc": knn_accuracy})

    # def backward(self, loss):
    #     pass

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.001, weight_decay=1e-4)
