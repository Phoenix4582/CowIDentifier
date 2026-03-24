import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import mutual_info_score

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

# Scipy Libraries
from scipy.optimize import linear_sum_assignment

# Use KNN to classify the embedding space
# Plot Confusion Matrix over predictions
def KNNAccuracy(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(train_embeddings, train_labels.ravel())
    # neigh.fit(train_embeddings, train_labels)

    # Total number of testing instances
    # total = len(test_labels.ravel()) - 1

    # Get the predictions from KNN
    predictions = neigh.predict(test_embeddings)

    # How many were correct?
    # correct = (predictions == test_labels.ravel()).sum()

    # Compute accuracy
    # accuracy = (float(correct) / total) * 100
    accuracy = neigh.score(test_embeddings, test_labels)

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

    # Calculate ARI score
    ari = adjusted_rand_score(test_labels.ravel(), predictions)

    return accuracy, precision, recall, f1

def KNNClusterPerformance(embeddings, labels, n_neighbors=5, return_pred_index=False):
    # Define the KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh.fit(embeddings, labels.ravel())
    predictions = neigh.predict(embeddings)

    if return_pred_index:
        mask = (predictions == labels)
        return np.array(mask, dtype='bool')

    else:
        total = len(labels.ravel())
        # How many were correct?
        correct = (predictions == labels.ravel()).sum()
        acc = (float(correct) / total) * 100
        acc_score = neigh.score(embeddings, labels) * 100
        precision, recall, f1_score = additional_metrics(predictions, labels.ravel())
        silhouette = silhouette_score(embeddings, labels)
        variance_ratio_criterion = calinski_harabasz_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)

        return acc, acc_score, silhouette, variance_ratio_criterion, db_score, precision, recall, f1_score

def KNNClusterConsistency(train_embeddings, train_labels, test_embeddings, test_labels, n_neighbors=5):
    # Define the KNN classifier
    neigh_train = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)
    # neigh_test = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-4)

    # Give it the embeddings and labels of the training set
    neigh_train.fit(train_embeddings, train_labels)
    # neigh_test.fit(train_embeddings, train_labels.ravel())

    lbls_pred = neigh_train.predict(test_embeddings)
    # print(test_labels)
    # print(lbls_pred)
    # print(lbls_pred == test_labels.ravel())
    # lbls_test = neigh_test.predict(test_embeddings)

    # col_choice = hungarian_algorithm(train_embeddings, test_embeddings, metric='cosine_distance')

    # lbls_pred_choice = np.array([lbls_pred[choice] for choice in col_choice])
    # assert lbls_pred.shape == lbls_pred_choice.shape

    total = len(test_labels.ravel())

    # How many were correct?
    correct = (lbls_pred == test_labels.ravel()).sum()
    # correct_choice = (lbls_pred_choice == test_labels.ravel()).sum()

    # Compute accuracy
    accuracy = (float(correct) / total) * 100
    accuracy_score = neigh_train.score(test_embeddings, test_labels) * 100

    # Extra metrics
    ari = adjusted_rand_score(test_labels, lbls_pred)
    mutual_info = mutual_info_score(test_labels, lbls_pred)

    return accuracy, accuracy_score, ari, mutual_info

def plot_confusion_matrix(preds, targets, dest:str):
    num_classes = len(np.unique(targets))
    # labels = [f"Cow {id+1}" for id in range(num_classes)]
    labels = [id for id in range(1, num_classes+1)]
    multi_cm = confusion_matrix(targets, preds, labels=labels, normalize='all')
    disp = ConfusionMatrixDisplay(confusion_matrix=multi_cm, display_labels=labels)
    disp.plot(include_values=False, cmap='plasma')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
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

def euclidean_distance(embd1, embd2):
    """
    Return Euclidean distance of two embeddings in numpy arrays.

    Input:
    - embd1, embd2: np.array() of N * 1, input arrays.
    Output:
    - distance
    """
    return np.linalg.norm(embd1 - embd2)

def cosine_distance(embd1, embd2):
    """
    Return Cosine similarity of two embeddings in numpy arrays.

    Input:
    - embd1, embd2: 1-D np.array() of size N, input arrays.
    Output:
    - distance
    """
    return np.dot(embd1.ravel(), embd2.ravel())/(np.linalg.norm(embd1) * np.linalg.norm(embd2))

def cost_matrix(embd_X, embd_Y, metric):
    """
    Return cost matrix between two embeddings under specified metric

    Input:
    - embd_X, embd_Y: np.array() of N1 * M, N2 * M, input arrays.
    - metric: function name to perform distance calculation
    Output:
    - cost matrix of size N1, N2
    """
    return np.array([[metric(x, y) for y in embd_Y] for x in embd_X])

def hungarian_algorithm(embd_X, embd_Y, metric:str):
    """
    Perform Linear Assignment on embd_X and embd_Y.
    Input:
    - embd_X, embd_Y: np.array() of N * M1, N * M2, input arrays.
    - metric: str, name of metric
    Output:
    - total cost after linear assignment(?)
    """
    # print(f"X embedding shape: {embd_X.shape}")
    # print(f"Y embedding shape: {embd_Y.shape}")

    cost = cost_matrix(embd_X, embd_Y, metric=eval(metric))
    # print(f"Cost shape:{cost.shape}")

    row, col = linear_sum_assignment(cost)
    # print(f"Linear Assignment total cost: {cost[row, col].sum()}")
    # print(col.shape)
    return col
