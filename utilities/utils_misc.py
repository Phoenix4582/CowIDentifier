import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Use KNN to classify the embedding space
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

def fetch_npz_data(path):
    with np.load(path) as data:
        return data['embeddings'], data['labels']
    # data = np.load(path)
    # # print(f"Successfully loaded npz file at {path}.")
    # return data['embeddings'], data['labels']
