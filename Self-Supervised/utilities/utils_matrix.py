import torch
# import torchmetrics
# from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(preds, targets, dest:str):
    num_classes = len(np.unique(targets))
    labels = [f"Cow {id+1}" for id in range(num_classes)]
    multi_cm = confusion_matrix(targets, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=multi_cm, display_labels=labels)
    disp.plot()

    plot.savefig(os.path.join(dest, "Confusion_Matrix.png"))

def plot_partial_confusion_matrix(preds, targets, partial: list, dest:str):
    num_classes = len(np.unique(targets))
    labels = [f"Cow {id+1}" for id in range(num_classes)]
    multi_cm = confusion_matrix(targets, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=multi_cm, display_labels=labels)
    disp.plot()

    plot.savefig(os.path.join(dest, "Confusion_Matrix_Partial.png"))
