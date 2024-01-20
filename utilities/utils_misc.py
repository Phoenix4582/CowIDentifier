import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

# Use KNN to classify the embedding space
# Plot Confusion Matrix over predictions
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels.ravel())

    # Total number of testing instances
    total = len(test_labels.ravel()) - 1

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    correct = (predictions == test_labels.ravel()).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100

    return accuracy

# Extend the old KNNAccuracy method to return additional metrics, i.e. mAP, mAR, etc.
def KNNMetrics(train_embeddings, train_labels, test_embeddings, test_labels, dest:str, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels.ravel())

    # Total number of testing instances
    total = len(test_labels.ravel()) - 1

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    correct = (predictions == test_labels.ravel()).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100

    # Plot confusion matrix
    plot_confusion_matrix(predictions, test_labels.ravel(), dest)

    # Calculate Precision, Recall, and F1 Score
    precision, recall, f1 = additional_metrics(predictions, test_labels.ravel())

    return accuracy, precision, recall, f1


def plot_confusion_matrix(preds, targets, dest:str):
    num_classes = len(np.unique(targets))
    # labels = [f"Cow {id+1}" for id in range(num_classes)]
    labels = [id for id in range(1, num_classes+1)]
    multi_cm = confusion_matrix(targets, preds, labels=labels, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=multi_cm, display_labels=labels).plot()
    plt.savefig(os.path.join(dest, "Confusion_Matrix.png"))

def additional_metrics(preds, targets, average='weighted'):
    num_classes = len(np.unique(targets))
    # labels = [f"Cow {id+1}" for id in range(num_classes)]
    labels = [id for id in range(1, num_classes+1)]
    precision = precision_score(preds, targets.ravel(), labels=labels, average=average)
    recall = recall_score(preds, targets.ravel(), labels=labels, average=average)
    f1 = f1_score(preds, targets.ravel(), labels=labels, average=average)
    return precision, recall, f1

def fetch_npz_data(path):
    with np.load(path) as data:
        return data['embeddings'], data['labels']
    # data = np.load(path)
    # # print(f"Successfully loaded npz file at {path}.")
    # return data['embeddings'], data['labels']
