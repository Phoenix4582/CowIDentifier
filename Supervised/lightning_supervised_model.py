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

from utilities.utils_misc import KNNAccuracy, KNNMetrics, additional_metrics, plot_confusion_matrix
from utilities.utils_tsne import init_tsne, init_umap, scatter
# from utilities.utils_matrix import plot_confusion_matrix

class MultiCamSupervisedModel(L.LightningModule):
   def __init__(self,
                backbone: str = 'ResNet18',
                hidden_dims: int = 64,
                num_classes: int = 90,
                lr: float = 0.001,
                imsize: int = 96,
                augment: bool = True,
                save_path: str = "outputs/Supervised/folder_name",
               ):
      super().__init__()
      self.imsize = imsize
      self.num_classes = num_classes
      self.augment = augment
      self.save_path = save_path

      self.pred = {}
      self.pred['train'] = []
      self.pred['test'] = []
      self.pred['val'] = []
      self.pred['predict'] = []
      self.labels= {}
      self.labels['train'] = []
      self.labels['test'] = []
      self.labels['val'] = []
      self.labels['predict'] = []

      # preset criteria variable for storing best training embeddings
      self.best_val_acc = 0

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
      self.net.fc = nn.Sequential(self.net.fc, nn.ReLU(), nn.Linear(1000, self.hparams.hidden_dims), nn.ReLU(), nn.Dropout(), nn.Linear(self.hparams.hidden_dims, self.num_classes))

   def doloss(self, batch, mode="train"):
      preds = self.net(batch[0])
      labels = batch[1] - 1 # CrossEntropyLoss need class id to be [0, num_classes).
      cmras = batch[2]
      # self.cmra[mode].append(cmras.clone().detach().cpu())

      lf = torch.nn.CrossEntropyLoss()
      labels = labels.reshape(-1)

      if self.augment:
         preds = self.net(self.augtrn(batch[0]))

      self.pred[mode].append(preds.clone().detach().cpu())
      self.labels[mode].append(labels.clone().detach().cpu())

      nll = lf(preds, labels)
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
      preds = torch.cat(self.pred[mode]).numpy()
      lbls = torch.cat(self.labels[mode]).numpy()
      total = len(lbls.ravel())
      preds = np.argmax(preds, axis=1)
      # print("Preds")
      # print(preds.shape)
      # print("Labels")
      # print(lbls.shape)
      correct = (preds == lbls.ravel()).sum()
      acc = (float(correct) / total) * 100
      self.log_dict({f"{mode}_acc": acc})
      precision, recall, f1 = additional_metrics(preds, lbls)
      matrix_path = os.path.join(self.save_path, mode)
      if not os.path.exists(matrix_path):
          os.makedirs(matrix_path)
      plot_confusion_matrix(preds, lbls, matrix_path)

      if mode != 'train':
          print(f"\n{mode.title()} Mode Accuracy: {acc} %")
          self.log_dict({f"{mode}_precision": precision})
          self.log_dict({f"{mode}_recall": recall})
          self.log_dict({f"{mode}_f1_score": f1})
          if mode == 'val' and acc > self.best_val_acc:
              self.best_val_acc = acc

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
      self.pred[mode].clear()
      self.labels[mode].clear()
      self.pred[mode]  = []
      self.labels[mode] = []

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
