import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pytorch_metric_learning import miners, losses
import torchvision.transforms as T
import torchvision
from torchvision.models import *
import lightning as L

from utilities.utils_misc import KNNAccuracy, KNNMetrics
from utilities.utils_tsne import init_tsne, init_umap, scatter
# from utilities.utils_matrix import plot_confusion_matrix

class MultiCamModel(L.LightningModule):
   def __init__(self,
                backbone: str = 'ResNet18',
                hidden_dims : int = 64,
                lossname: str = 'NTXentLoss()',
                lr: float = 0.001,
                imsize: int = 96,
                mining : bool = True,
                augment : bool = True,
                save_path: str = "outputs/folder_name",
                perplexity: int = 25,
                u_neighbors: int = 25,
                check_cameras: bool = True,
               ):
      super().__init__()
      self.lossname = lossname
      self.imsize = imsize
      self.mining = mining
      self.augment = augment
      self.save_path = save_path

      self.embed = {}
      self.embed['train'] = []
      self.embed['test'] = []
      self.embed['val'] = []
      self.embed['predict'] = []
      self.labels= {}
      self.labels['train'] = []
      self.labels['test'] = []
      self.labels['val'] = []
      self.labels['predict'] = []
      self.cmra = {}
      self.cmra['train'] = []
      self.cmra['test'] = []
      self.cmra['val'] = []
      self.cmra['predict'] = []

      # preset criteria variable for storing best training embeddings
      self.best_val_acc = 0

      # initialise tsne and umap
      self.tsne = init_tsne(perplexity=perplexity)
      self.ump = init_umap(n_neighbors=u_neighbors)

      # Flag for mono/multi-camera mode
      self.check_cameras = check_cameras

      if augment:
         self.augtrn = T.Compose([T.RandomResizedCrop(self.imsize, scale=(0.95, 1.0), ratio=(0.95,1.05)),
                                  T.Resize(self.imsize),
                                  T.ElasticTransform(alpha=100.0),
                                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
      self.save_hyperparameters()
      ntnme  = backbone+'_Weights.DEFAULT'
      ldr = f"self.net = {backbone.lower()}(weights='{ntnme}')"
      print(ldr)
      exec(ldr)
      self.net.fc = nn.Sequential(self.net.fc, nn.ReLU(), nn.Linear(1000, self.hparams.hidden_dims))

   def doloss(self, batch, mode="train"):
      embeddings = self.net(batch[0])
      labels = batch[1]
      cmras = batch[2]

      self.embed[mode].append(embeddings.clone().detach().cpu())
      self.labels[mode].append(labels.clone().detach().cpu())
      self.cmra[mode].append(cmras.clone().detach().cpu())

      lf = eval(f'losses.{self.lossname}')
      hard_pairs = None
      # if mode == 'val':
      #     print(f"Labels: {labels}")
      # labels = labels.squeeze()
      labels = labels.reshape(-1)

      if self.augment:
         augembedd = self.net(self.augtrn(batch[0]))
         embeddings = torch.cat([embeddings, augembedd])
         labels = torch.cat([labels, labels])
         cmras = torch.cat([cmras, cmras])
      if self.mining:
         miner = miners.MultiSimilarityMiner()
         hard_pairs = miner(embeddings, labels)

      if hard_pairs is not None:
         nll = lf(embeddings, labels, hard_pairs)
      else:
         nll = lf(embeddings, labels)

      if not mode=='predict':
         self.log(mode + "_loss", nll, prog_bar=True)
      return nll

   def training_step(self, batch, batch_idx):
      return self.doloss(batch, mode='train')

   def validation_step(self, batch, batch_idx):
      return self.doloss(batch, mode='val')

   def test_step(self, batch, batch_idx):
      return self.doloss(batch, mode='test')

   # def predict_step(self, batch, batch_idx):
   #    return self.doloss(batch, mode='predict')


   def _epoch_end(self, mode='train'):
      # assert mode == 'train' or mode == 'val'
      embd = torch.cat(self.embed[mode]).numpy()
      lbls = torch.cat(self.labels[mode]).numpy()
      cmr  = torch.cat(self.cmra[mode]).numpy()

      tsne_embed = self.tsne.fit(embd)
      ump_embed = self.ump.fit_transform(embd)
      
      scatter(tsne_embed, lbls, f"TSNE-{mode}-label", os.path.join(self.save_path, mode, "tsne_label", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")
      scatter(tsne_embed, cmr, f"TSNE-{mode}-camera", os.path.join(self.save_path, mode, "tsne_camera", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")

      if self.check_cameras:
         scatter(ump_embed, lbls, f"UMAP-{mode}-label", os.path.join(self.save_path, mode, "umap_label", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")
         scatter(ump_embed, cmr, f"UMAP-{mode}-camera", os.path.join(self.save_path, mode, "umap_camera", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")

      if mode == 'train' or mode == 'val':
          train_embd = torch.cat(self.embed['train']).numpy()
          train_lbls = torch.cat(self.labels['train']).numpy()
          show_matrix = False
          knn_accuracy = KNNAccuracy(train_embd, train_lbls, embd, lbls)
      else:
          source = os.path.join(self.save_path, "train_embed_data_best.npz")
          train_data = np.load(source, allow_pickle=True)
          train_embd = train_data["embeddings"]
          train_lbls = train_data["labels"]
          knn_accuracy, precision, recall, f1 = KNNMetrics(train_embd, train_lbls, embd, lbls, os.path.join(self.save_path, mode))

      if mode != 'train':
          print(f"\n{mode.title()} Mode KNNAccuracy: {knn_accuracy} %")
          if mode == 'val' and knn_accuracy > self.best_val_acc:
              dest = os.path.join(self.save_path, "train_embed_data_best.npz")
              np.savez(dest, embeddings=train_embd, labels=train_lbls)
              self.best_val_acc = knn_accuracy

      self.log_dict({f"{mode}_acc": knn_accuracy})
      if mode == 'test':
          self.log_dict({f"{mode}_precision": precision})
          self.log_dict({f"{mode}_recall": recall})
          self.log_dict({f"{mode}_f1_score": f1})


   def on_train_epoch_end(self):
      self._epoch_end(mode='train')

   def on_validation_epoch_end(self):
      # if self.current_epoch == 0:
      #     return
      self._epoch_end(mode='val')

   def on_test_epoch_end(self):
      self._epoch_end(mode='test')
      print('\nEnd of test_epoch\n')

   def on_predict_epoch_end(self):
       pass

   def _epoch_start(self, mode):
      self.embed[mode].clear()
      self.labels[mode].clear()
      self.cmra[mode].clear()
      self.embed[mode]  = []
      self.labels[mode] = []
      self.cmra[mode]  = []

   def on_train_epoch_start(self):
      self._epoch_start('train')

   def on_validation_epoch_start(self):
      self._epoch_start('val')

   def on_test_epoch_start(self):
      print('\nStart of test_epoch\n')
      self._epoch_start('test')

   # def on_predict_epoch_start(self):
   #    print('\nStart of predict_epoch\n')
   #    self._epoch_start('predict')

   def configure_optimizers(self):
      return torch.optim.AdamW(self.parameters(), lr=0.01)
