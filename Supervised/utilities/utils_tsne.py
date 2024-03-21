import os
import numpy as np
from PIL import Image

# MatPlotLib and Sklearn libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

# from utilities.visualise_embeddings import scatter
import openTSNE
import umap.umap_ as umap

# Define our own plot function
def scatter(x, labels, title=None, filename=None, file_format="pdf"):
    # Get the number of classes (number of unique labels)
    num_classes = np.unique(labels).shape[0]

    # Choose a color palette with seaborn.
    # HLS, MAGMA, ROCKET, MAKO, VIRIDIS, CUBEHELIX
    palette = np.array(sns.color_palette("hls", num_classes+1))

    # Map the colours to different labels
    label_colours = np.array([palette[int(labels[i])] for i in range(labels.shape[0])])

    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    # Set title of image if have one
    if title is not None:
        ax.set_title(title)

    # Plot the points
    ax.scatter(    x[:,0], x[:,1],
                lw=0, s=10,
                c=label_colours,
                marker="o")

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.tight_layout()

    # Save it to file
    if filename is not None:
        # print(filename)
        filename_parent = "/".join(filename.split("/")[:-1])
        if not os.path.exists(filename_parent):
            os.makedirs(filename_parent)
        plt.savefig(filename + "." + file_format)

    plt.close()

def plot_results(X, Y_, means, covariances, title, filename, file_format="pdf"):
    # Get the number of classes (number of unique labels)
    num_classes = np.unique(Y_).shape[0]

    # Choose a color palette with seaborn.
    # HLS, MAGMA, ROCKET, MAKO, VIRIDIS, CUBEHELIX
    palette = np.array(sns.color_palette("hls", num_classes+1))

    # Map the colours to different labels
    label_colours = np.array([palette[int(Y_[i])] for i in range(Y_.shape[0])])

    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.set_title(title)

    for i, (mean, covar, color) in enumerate(zip(means, covariances, label_colours)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=10, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / (u[0] + 1e-8))
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.tight_layout()

    # Save it to file
    if filename is not None:
        # print(filename)
        filename_parent = "/".join(filename.split("/")[:-1])
        if not os.path.exists(filename_parent):
            os.makedirs(filename_parent)
        plt.savefig(filename + "." + file_format)

    plt.close()

def create_gif(source:str, dest_folder:str, name:str, format:str = "png", delay:int = 5):
    """
    Create a gif from a folder of images with delay at start and end.
    """
    images = [Image.open(os.path.join(source,im)) for im in os.listdir(source) if im.endswith("."+format)]
    images_start_delay = [images[0] for _ in range(delay)]
    images_end_delay = [images[-1] for _ in range(delay)]
    images = images_start_delay + images + images_end_delay

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    gif_name = name + ".gif"
    images[0].save(os.path.join(dest_folder, gif_name), save_all=True, append_images=images[1:], optimize=False, duration=30)

    # # Perform visualisation for embeddings during training, save images for gif generation
    # def prepare_gif(self, epoch, title_str):
    #     npz_files = [ele for ele in os.listdir(self.args.fold_out_path) if ele.endswith("embeddings.npz")]
    #     for npz_file in npz_files:
    #         file_type = npz_file[:-15]
    #         # file_type = npz_file.strip("_embeddings.npz")
    #         file_name = os.path.join(self.args.fold_out_path, "gifs", file_type, f"record_{str(epoch).zfill(4)}")
    #         run_commands  = f"python3 utilities/visualise_embeddings.py"
    #         run_commands += f" --embeddings_file={os.path.join(self.args.fold_out_path, npz_file)}"
    #         run_commands += f" --open_tsne"
    #         run_commands += f" --title={title_str}"
    #         run_commands += f" --file_name={file_name}"
    #         run_commands += f" --format=png"
    #
    #         subprocess.call([run_commands], shell=True)

# Perform visualisation for embeddings during training, save images for gif generation
def prepare_vis(epoch, num_classes, fold_out_base, tsne, title_str, format="png"):
    fold_out_parent = "/".join(fold_out_base.split("/")[:-1])
    fold_out_path = fold_out_base
    npz_files = [ele for ele in os.listdir(fold_out_path) if ele.endswith("embeddings.npz")]
    for npz_file in npz_files:
        file_type = npz_file[:-15]
        # file_type = npz_file.strip("_embeddings.npz")
        file_name_label = os.path.join(fold_out_parent, "vis", file_type, "label", f"record_{str(epoch).zfill(4)}")
        file_name_camera = os.path.join(fold_out_parent, "vis", file_type, "camera", f"record_{str(epoch).zfill(4)}")

        embeddings_file = os.path.join(fold_out_path, npz_file)
        embeddings = np.load(embeddings_file)
        reduction = tsne.fit(embeddings['embeddings'])

        scatter(reduction, embeddings['labels'], title_str, file_name_label, format)
        scatter(reduction, embeddings['camera'], title_str, file_name_camera, format)


def prepare_kmeans(epoch, num_classes, fold_out_base, grouper, title_str, format="png"):
    fold_out_parent = "/".join(fold_out_base.split("/")[:-1])
    fold_out_path = fold_out_base
    npz_files = [ele for ele in os.listdir(fold_out_path) if ele.endswith("embeddings.npz")]
    for npz_file in npz_files:
        file_type = npz_file[:-15]
        file_name_cluster = os.path.join(fold_out_parent, "vis", file_type, "cluster_kmeans", f"record_{str(epoch).zfill(4)}")

        embeddings_file = os.path.join(fold_out_path, npz_file)
        embeddings = np.load(embeddings_file)

        # Plot KMeans over embeddings
        embedding_groups = grouper.fit_predict(embeddings['embeddings'])
        scatter(embeddings['embeddings'], grouper.labels_, title_str, file_name_cluster, format)


def prepare_gmm(epoch, num_classes, fold_out_base, gmm, title_str, format="png"):
    fold_out_parent = "/".join(fold_out_base.split("/")[:-1])
    fold_out_path = fold_out_base
    npz_files = [ele for ele in os.listdir(fold_out_path) if ele.endswith("embeddings.npz")]
    for npz_file in npz_files:
        file_type = npz_file[:-15]
        file_name_gmm = os.path.join(fold_out_parent, "vis", file_type, "cluster_gmm", f"record_{str(epoch).zfill(4)}")

        embeddings_file = os.path.join(fold_out_path, npz_file)
        embeddings = np.load(embeddings_file)

        # Plot Gaussian Mixture Model over embeddings
        gmm = gmm.fit(embeddings['embeddings'])
        plot_results(embeddings['embeddings'], gmm.predict(embeddings['embeddings']), gmm.means_, gmm.covariances_, title_str, file_name_gmm, format)


def init_tsne(n_components=2, perplexity=25):
    """
    Initialise a TSNE model with given parameters from OpenTSNE library.
    """
    tsne = openTSNE.TSNE(n_components=n_components,
                         perplexity=perplexity,
                         metric="cosine",
                         n_jobs=8,
                         random_state=42,
                         verbose=False,)

    return tsne

def init_umap(n_components=2, n_neighbors=25):
    """
    Initialise a UMAP object from UMAP library.
    """
    ump = umap.UMAP(n_components=n_components,
                     n_neighbors=n_neighbors,
                     metric="cosine",
                     verbose=False,)

    return ump
