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

from utilities.utils_misc import KNNClusterConsistency, KNNClusterPerformance
from utilities.utils_tsne import init_tsne, init_umap, scatter, scatter_highlight
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
               ):
      super().__init__()
      self.lossname = lossname
      self.imsize = imsize
      self.mining = mining
      self.augment = augment
      self.save_path = save_path

      self.embed = {}
      self.embed['train'] = []
      self.embed['val'] = []
      self.embed['test_base'] = []
      self.embed['test_target'] = []
      self.embed['predict'] = []
      self.labels= {}
      self.labels['train'] = []
      self.labels['val'] = []
      self.labels['test_base'] = []
      self.labels['test_target'] = []
      self.labels['predict'] = []
      # self.cmra = {}
      # self.cmra['train'] = []
      # self.cmra['test'] = []
      # self.cmra['val'] = []
      # self.cmra['predict'] = []

      # # preset criteria variable for storing best training embeddings
      # self.min_val_loss = 0

      # initialise tsne and umap
      self.tsne = init_tsne()
      self.ump = init_umap()

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
      # cmras = batch[2]

      self.embed[mode].append(embeddings.clone().detach().cpu())
      self.labels[mode].append(labels.clone().detach().cpu())

      lf = eval(f'losses.{self.lossname}')
      hard_pairs = None
      labels = labels.reshape(-1)

      if self.augment and mode == "train":
         augembedd = self.net(self.augtrn(batch[0]))
         embeddings = torch.cat([embeddings, augembedd])
         labels = torch.cat([labels, labels])
      if self.mining and mode == "train":
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

   def test_step(self, batch, batch_idx, dataloader_idx):
      assert dataloader_idx < 2
      if dataloader_idx == 0:
          loss = self.doloss(batch, mode='test_base')
      else:
          loss = self.doloss(batch, mode='test_target')

      return loss

      # return self.doloss(batch, mode='test')

   # def predict_step(self, batch, batch_idx):
   #    return self.doloss(batch, mode='predict')


   def _epoch_end(self, mode='train'):
      # assert mode == 'train' or mode == 'val'
      if mode == 'train' or mode == 'val':
          embd = torch.cat(self.embed[mode]).numpy()
          lbls = torch.cat(self.labels[mode]).numpy()
          # cmr  = torch.cat(self.cmra[mode]).numpy()

          tsne = init_tsne()
          tsne_embed = tsne.fit(embd)
          scatter(tsne_embed, lbls, f"TSNE-{mode}-label", os.path.join(self.save_path, mode, "tsne_label", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")
          # scatter(tsne_embed, cmr, f"TSNE-{mode}-camera", os.path.join(self.save_path, mode, "tsne_camera", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")

          ump = init_umap()
          ump_embed = ump.fit_transform(embd)
          scatter(ump_embed, lbls, f"UMAP-{mode}-label", os.path.join(self.save_path, mode, "umap_label", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")
          # scatter(ump_embed, cmr, f"UMAP-{mode}-camera", os.path.join(self.save_path, mode, "umap_camera", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")

          cluster_performance, silhouette, vrc, dbs = KNNClusterPerformance(embd, lbls)
          print(f"\n{mode.title()} Mode Cluster Performance: {cluster_performance*100:.3f}%")
          print(f"{mode.title()} Mode Silhouette Score: {silhouette:.5f}")
          print(f"{mode.title()} Mode Variance Ratio Criterion: {vrc:.5f}")
          print(f"{mode.title()} Mode Davies-Bouldin score: {dbs:.5f}")

      elif mode == 'test':
          embd_base = torch.cat(self.embed[f'{mode}_base']).numpy()
          lbls_base = torch.cat(self.labels[f'{mode}_base']).numpy()
          embd_target = torch.cat(self.embed[f'{mode}_target']).numpy()
          lbls_target = torch.cat(self.labels[f'{mode}_target']).numpy()
          print(f"\n{mode.title()} Mode")
          for sub_name in ['base', 'target']:
              tsne = init_tsne()
              ump = init_umap()
              embd = embd_base if sub_name == 'base' else embd_target
              lbls = lbls_base if sub_name == 'base' else lbls_target
              mask = KNNClusterPerformance(embd, lbls, return_pred_index=True)
              tsne_embed = tsne.fit(embd)
              scatter(tsne_embed, lbls, f"TSNE-{mode}-label-{sub_name}", os.path.join(self.save_path, f"{mode}_{sub_name}", "tsne_label", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")
              scatter_highlight(tsne_embed, lbls, mask, f"TSNE-{mode}-label-{sub_name}", os.path.join(self.save_path, f"{mode}_{sub_name}", "tsne_label", f"record-{str(self.current_epoch).zfill(5)}-highlight"), file_format="png")
              ump_embed = ump.fit_transform(embd)
              scatter(ump_embed, lbls, f"UMAP-{mode}-label-base-{sub_name}", os.path.join(self.save_path, f"{mode}_{sub_name}", "umap_label", f"record-{str(self.current_epoch).zfill(5)}"), file_format="png")
              scatter_highlight(ump_embed, lbls, mask, f"UMAP-{mode}-label-base-{sub_name}", os.path.join(self.save_path, f"{mode}_{sub_name}", "umap_label", f"record-{str(self.current_epoch).zfill(5)}-highlight"), file_format="png")
              cluster_performance, silhouette, vrc, dbs = KNNClusterPerformance(embd, lbls)
              print(f"\n{sub_name.title()} Mode Cluster Performance: {cluster_performance*100:.3f}%")
              print(f"{sub_name.title()} Mode Silhouette Score: {silhouette:.5f}")
              print(f"{sub_name.title()} Mode Variance Ratio Criterion: {vrc:.5f}")
              print(f"{sub_name.title()} Mode Davies-Bouldin score: {dbs:.5f}")

          acc, acc_c, cluster_ari, mutual = KNNClusterConsistency(embd_base, lbls_base, embd_target, lbls_target)
          print(f"\n{mode.title()} Mode Raw Accuracy: {acc:.5f}%", flush=True)
          print(f"{mode.title()} Mode Scored Accuracy: {acc_c:.5f}%", flush=True)
          print(f"{mode.title()} Mode Cluster Consistency(ARI): {cluster_ari:.5f}", flush=True)
          print(f"{mode.title()} Mode Mutual Information: {mutual:.5f}", flush=True)
          self.log_dict({f"{mode}_acc": acc})
          self.log_dict({f"{mode}_acc_c": acc_c})
          self.log_dict({f"{mode}_ari": cluster_ari})
          self.log_dict({f"{mode}_mutual": mutual})
      # if mode == 'train' or mode == 'val':
      #     train_embd = torch.cat(self.embed['train']).numpy()
      #     train_lbls = torch.cat(self.labels['train']).numpy()
      #     # acc, acc_c, cluster_ari, mutual = KNNClusterConsistency(train_embd, train_lbls, embd, lbls)
      #
      # else:
      #     source = os.path.join(self.save_path, "train_embed_data_best.npz")
      #     train_data = np.load(source, allow_pickle=True)
      #     train_embd = train_data["embeddings"]
      #     train_lbls = train_data["labels"]
      #     acc, acc_c, cluster_ari, mutual = KNNClusterConsistency(train_embd, train_lbls, embd, lbls)

          # knn_accuracy, precision, recall, f1 = KNNMetrics(train_embd, train_lbls, embd, lbls, os.path.join(self.save_path, mode))

      # if mode != 'train':
      #     # print(f"\n{mode.title()} Linear Assignment cost: {cost}")
      #     if mode == 'val' and val_loss < self.min_val_loss:
      #         dest = os.path.join(self.save_path, "train_embed_data_best.npz")
      #         np.savez(dest, embeddings=train_embd, labels=train_lbls)
      #         self.min_val_loss = val_loss

          # if mode != 'val':
          #     print(f"\n{mode.title()} Mode Raw Accuracy: {acc:.5f}%", flush=True)
          #     print(f"{mode.title()} Mode Scored Accuracy: {acc_c:.5f}%", flush=True)
          #     print(f"{mode.title()} Mode Cluster Consistency(ARI): {cluster_ari:.5f}", flush=True)
          #     print(f"{mode.title()} Mode Mutual Information: {mutual:.5f}", flush=True)
          #     self.log_dict({f"{mode}_acc": acc})
          #     self.log_dict({f"{mode}_acc_c": acc_c})
          #     self.log_dict({f"{mode}_ari": cluster_ari})
          #     self.log_dict({f"{mode}_mutual": mutual})

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
      if mode == 'test':
          self.embed[f'{mode}_base'].clear()
          self.labels[f'{mode}_base'].clear()
          self.embed[f'{mode}_base']  = []
          self.labels[f'{mode}_base'] = []
          self.embed[f'{mode}_target'].clear()
          self.labels[f'{mode}_target'].clear()
          self.embed[f'{mode}_target']  = []
          self.labels[f'{mode}_target'] = []
      else:
          self.embed[mode].clear()
          self.labels[mode].clear()
          self.embed[mode]  = []
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
