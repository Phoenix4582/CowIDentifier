# Core libraries
import os, sys, csv
import argparse, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import matplotlib as mpl
import matplotlib.colors as mcolors
import datetime

# to disable colour error
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

# Local libraries
from utils_visual import ACC, makedirs, toplabels, count_if_enough
from tsne_embedding import scatter_singel, scatter_density, scatter_images, scatter_overlap
mpl.use('Agg')

size_dot = 10
sizeHigh = 20

def get_date_now():
    now = datetime.datetime.now()
    return f"{now.month}-{now.day}-{now.year}"

# Define our own plot function
def scatter_color_rainbow(reduction, select_reduction, cm, fn2, non_zero_likely=[0], alpha=1):
    if len(non_zero_likely) > 1:
        plt.figure(figsize=(10, 8))
    else:
        plt.figure(figsize=(8, 8))

    plt.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=size_dot, c=[0.8, 0.8, 0.8], marker="o")
    if len(non_zero_likely) > 1:
        plt.scatter(select_reduction[:, 0], select_reduction[:, 1], c=np.reshape(non_zero_likely,-1), lw=0, s=25,
                    cmap=cm, marker="o", vmin=min(non_zero_likely), vmax=max(non_zero_likely), alpha=alpha)
        cb = plt.colorbar(shrink=0.9)
        cb.ax.tick_params(labelsize=16)
        cb.set_label('Likelihood', size=16)
    else:
        plt.scatter(select_reduction[:, 0], select_reduction[:, 1], lw=0, s=size_dot, c='deepskyblue', marker="o")

    plt.axis('off')
    plt.axis('tight')
    plt.tight_layout()

    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, fn2 + ".pdf"))
    plt.savefig(os.path.join(args.out_path, fn2 + ".png"))
    plt.show()
    plt.close()


def scatter_p(x, labels, filename, highlight=False):
    # Get the number of classes (number of unique labels)
    # num_classes = np.unique(labels).shape[0]
    # Choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 600))
    # Randomly shuffle with the same seed
    np.random.seed(26)
    np.random.shuffle(palette)

    # Convert labels to int
    labels = labels.astype(int)

    # Map the colours to different labels
    label_colours = np.array([palette[labels[i]] for i in range(labels.shape[0])])
    for i in range(len(labels)):
        if labels[i] < 0:
            label_colours[i] = [0, 0, 0]

    if 'xxx' in filename:
        return label_colours
    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # Do we want to highlight some particular (e.g. difficult) labels
    if highlight:
        # Which labels should we highlight (the "difficult" individuals)
        highlight_labels = [54, 69, 73, 173]
        # Colour for non-highlighted points
        label_colours = np.zeros(label_colours.shape)
        # Alpha value for non-highlighted points
        alpha = 1.0
        # Plot all the points with some transparency
        ax.scatter(x[:, 0], x[:, 1], lw=0, s=10, c=label_colours, marker="o", alpha=alpha)
        # Highlighted points
        h_pts = np.array([x[i, :] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
        # Colours
        h_colours = np.array([palette[labels[i]] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
        # There may not have been any occurences of that label
        if h_pts.size != 0:
            # Replot highlight points with no alpha
            ax.scatter(h_pts[:, 0], h_pts[:, 1], lw=0, s=10, c=h_colours, marker="o")
        else:
            print(f"Didn't find any embeddings with the label: {highlight_labels}")
    # Just colour each point normally
    else:
        # Plot the points
        ax.scatter(x[:, 0], x[:, 1], lw=0, s=10, c=label_colours, marker="o",alpha=0.6)

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.tight_layout()
    # Save it to file
    # plt.show()
    if not args.png_only:
        plt.savefig(filename + ".pdf")
    plt.savefig(filename + ".png")
    plt.close()
    return label_colours, palette


# Load and visualise embeddings via t-SNE

def plotEmbeddings_tsneMul(args, emd_path):

    # Ensure there's something there
    if not os.path.exists(args.embeddings_file):
        print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
        sys.exit(1)

    # Load the embeddings into memory
    embeddings = np.load(emd_path)
    # Visualise the learned embedding via t-SNE
    visualiser = TSNE(n_components=2, perplexity=args.perplexity)
    # 2d data after TSNE
    reduction = visualiser.fit_transform(embeddings['embeddings'])

    # Plot the results and save to file
    # corrected knn label
    name_knn = 'xxx'  # discard knn labels
    e_lb = embeddings['labels_folder']
    # if np.max(embeddings['labels_folder']) > args.train_eb:  # for the train_eb big code
    #     # match gt to img
    #     csv_path = os.path.join(args.embeddings_file[:-10], 'correct.csv')
    #     csvFile = csv.reader(open(csv_path, 'r'), delimiter=',')
    #     reader = list(csvFile)
    #     del reader[0]
    #     ap_gt = []
    #     ap_fold = []
    #     for i in range(len(reader) - 1):
    #         ap_fold.append(reader[i][1])
    #         if reader[i][3] == '1':
    #             ap_gt.append((int(float(reader[i][4]))))
    #         else:
    #             ap_gt.append((int(float(reader[i][2]))))
    #
    #     # assign gt
    #     e_lb = []
    #     for i, item in enumerate(embeddings['path']):
    #         folder = item[:5]
    #         pos = ap_fold.index(folder)
    #         e_lb.append(ap_gt[pos])
    #     print(f'The right class is {len(set(e_lb))}')
    #
    #     e_lb = np.array(e_lb)
    keepX = list(range(0, len(reduction)))

    # else:
    #     # Dataset: start from 5 Feb, end in March 11. 30 days (Feb: No 6,7,8,9,17,30)
    #     date = args.date
    #     keepX = []
    #     for i, item in enumerate(embeddings['path']):
    #         pos = item.find('2020-')
    #         month_ = int(item[pos + 5:pos + 7])
    #         date_ = int(item[pos + 8:pos + 10])
    #         if (date[0] <= month_) * (month_ <= date[2]) * (date[1] <= date_) * (date_ <= date[3]):
    #             keepX.append(i)

    tmplabels_knn = e_lb
    # return label colors only
    label_colours = scatter_p(reduction[keepX], tmplabels_knn[keepX], os.path.join(args.out_path, name_knn))
    # label from folder name
    label_colours_model, palette = scatter_p(reduction[keepX], e_lb[keepX],
                                             os.path.join(args.out_path, 'Folder_label on tsne'))
    return reduction[keepX], embeddings['embeddings'][keepX], label_colours, tmplabels_knn[keepX], label_colours_model, \
           e_lb[keepX], embeddings['path'][keepX], palette



def PlotGMM(gmm, X, reduction, args, path, ax=None):
    '''
    return labels, log every hilikly hood spot and their top2 label [4,2] , colour of hilighted , second top label
    '''

    ax = ax or plt.gca()
    gmm.fit(X)
    PreLabelGMM = gmm.predict(X)  # get prediction label

    if args.train_eb> 20:
        _, _ = scatter_p(reduction, PreLabelGMM, os.path.join(args.out_path, 'GMM-zFirstPredictColor'))

    # find the top points has high prob in 2 clusters == high uncertainty
    prob_list = gmm._estimate_weighted_log_prob(X)  # do not norm using logsumexp like the original code
    findTop2 = np.sort(prob_list)  # sort to get dots with the biggest 2 probability
    top_1 = findTop2[:, -1] # each point belongs to which cluster most likely
    top_2 = findTop2[:, -2] # each point belongs to the second most likely

    # Rank top1-top2
    RankLikely = []
    for i in range(0, len(top_1)):
        RankLikely.append(round(top_1[i] - top_2[i], 4))

    RankLikely = sorted(enumerate(RankLikely), key=lambda e: e[1])  # index and value

    # assign the sorted to top and bottom
    index_bottom = []
    index_top = []
    for i, item in enumerate(RankLikely):
        if i < int(len(RankLikely) / 2):  # if small, top uncertainty
            index_top.append(item[0])
        else:
            index_bottom.append(item[0])
    #assert len(index_bottom) == len(index_top)

    # find the path(image name) and save the path
    #index_revolt = index_top[:len(index_top)] + index_bottom[:int(len(index_top) / 2)]  # account for 75 percent
    index_revolt = index_top[:len(index_top)] + index_bottom[:int(len(index_top))] # all points.
    # revolt_path = path[index_revolt]
    # np.savez(args.out_path + "/top_bottom.npz", top_path=path[index_top], bottom_path=path[index_bottom], revolt_path=revolt_path)

    top_label = toplabels(PreLabelGMM, prob_list, args.topN)

    top_label_exchange = []
    colour_va = []  # top1-top2 to express color
    interaction_2 = top_label[:, :2].tolist()
    for i in range(0, len(RankLikely)):
        if i in index_revolt:
            colour_va.append(int(top_1[i] - top_2[i]))
            top_label_exchange.append(interaction_2[i])
        else:
            colour_va.append(0)

    SaveEnough = count_if_enough(top_label_exchange, args)  # exchangable labels if the number is enough

    return PreLabelGMM, SaveEnough, colour_va, top_label, RankLikely


def plot4image(reduction, select_reduction_md, GT_LabelColor, label_colours_gmm_md,
               select_label_colours_gmm_md, select_label_coloursanot_md, acc2, ARI_Score, args, name):
    # plot big picture
    scatter_color_rainbow(reduction, select_reduction_md, select_label_coloursanot_md, 'differ' + name)

    fig = plt.figure(figsize=(8, 8))
    ax3 = fig.add_subplot(221)
    if args.show_title:
        plt.title(name)
    ax3.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=GT_LabelColor, marker="o")
    ax3.axis('off')

    # not important
    ax6 = fig.add_subplot(223)
    # plt.title("GMM-Tsne-2d")
    # PlotGMM(gmm, reduction, ax=ax6)
    # ax6.axis('off')
    if args.show_title:
        plt.title("GMM-Correct")
    ax6.axis('off')
    ax6.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=label_colours_gmm_md, marker="o")

    ax7 = fig.add_subplot(224)
    if args.show_title:
        plt.title("Highlight Difference")  # originan   c=label_colours_gmm
    ax7.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax7.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=5, c=select_label_colours_gmm_md,
                marker="o")
    ax7.axis('off')

    ax8 = fig.add_subplot(222)
    ax8.scatter(reduction[:, 0], reduction[:, 1], lw=0, s=5, c=[0.8, 0.8, 0.8], marker="o")
    ax8.scatter(select_reduction_md[:, 0], select_reduction_md[:, 1], lw=0, s=5, c=select_label_coloursanot_md,
                marker="o")
    ax8.axis('off')

    plt.suptitle("Set {} Classes. Acc: {} ARI: {}".format(args.components, acc2, ARI_Score))
    plt.axis('tight')
    plt.tight_layout()
    if not args.png_only:
        plt.savefig(os.path.join(args.out_path, "All_folder.pdf"))
    plt.savefig(os.path.join(args.out_path, "All_folder.png"))
    #if os.path.exists("/home/io18230/Desktop"):
    plt.show()
    plt.close()

    # ****************************"Plotting all in one"*********************************#


def acc_plotall(GTLabel, GTcolor, PreLabelGMM, col, label_topN, top_2_exchange_label, RankLikely, args, path, name=''):
    # ari
    ARI_Score = round(metrics.adjusted_rand_score(PreLabelGMM, GTLabel), 3)
    print(f'ARI = {ARI_Score}')

    feb_pre, difference_label, acc, lover_label, acc_top2, GrayPoints = ACC(GTLabel, PreLabelGMM, top_2_exchange_label,
                                                                            label_topN, args)

    path_tmp = os.path.join(args.out_path, os.path.basename(args.embeddings_file)[:-4])
    # scatter_overlap(reduction, feb_pre, path_tmp, palette=palette, show_class=0, name='gmm')
    # scatter_density(reduction, feb_pre, path_tmp, path, args, palette=palette, name='_Gmm', show_class = 1, textup=4, imgSize = 30)
    scatter_density(reduction, X_128, feb_pre, path_tmp, col, label_topN, path, args, outlier_mask = (['a']), palette=palette, name='_Gmm',
                    textupp=4, imgSize=30) #???

    if args.single_highlight:
        singleImag(reduction, feb_pre, args, palette, path, name='Single_Gmm') # defult=1 position 0 dots 1 images 2 image on the side
        singleImag(reduction, feb_pre, args, palette, path, name='Single_Gp', position=2)

    label_colours_gmm_fd, _ = scatter_p(reduction, feb_pre, os.path.join(args.out_path, name))

    # print CHANGE COLOUR NAMES of different labels between gt and prediciton -- select-
    highlight_reduction = np.zeros(shape=(np.count_nonzero(col), 2))
    non_zero_likely = np.zeros(shape=(np.count_nonzero(col), 1))
    i = 0
    for count, item in enumerate(col):
        if item != 0:
            highlight_reduction[i] = reduction[count]
            non_zero_likely[i] = col[count]
            i = i + 1


    # the last part
    select_reduction = np.zeros(shape=(len(difference_label), 2))
    select_label_colours_gmm_fd = np.zeros(shape=(len(difference_label), 3))
    select_label_coloursanothre = np.zeros(shape=(len(difference_label), 3))
    for i, item in enumerate(difference_label):
        select_reduction[i] = reduction[item]
        select_label_colours_gmm_fd[i] = label_colours_gmm_fd[item]
        select_label_coloursanothre[i] = GTcolor[item]

    if name == 'GMM-CKNN-Label on tsne':
        plot4image(reduction, select_reduction, GTcolor, label_colours_gmm_fd,
                   select_label_colours_gmm_fd, select_label_coloursanothre, acc, ARI_Score, args, name='KNN')
    if name == 'GMM-folder-Label on tsne':
        plot4image(reduction, select_reduction, GTcolor, label_colours_gmm_fd,
                   select_label_colours_gmm_fd, select_label_coloursanothre, acc, ARI_Score, args, name='Folder')

    return feb_pre, acc, ARI_Score


def singleImag(reduction, label, args, palette, path, name, position=1):
    makedirs(os.path.join(args.out_path, name))
    path_tmp = os.path.join(args.out_path, name, os.path.basename(args.embeddings_file)[:-4])
    x, y, l = scatter_singel(reduction, label, path_tmp, args, palette=palette, img_path=path,
                             position=position)  # position 0 dots 1 images 2 image on the side

    # histogram
    if name == 'Single_gt':
        xxx = list(range(len(x)))
        total_width, n = 0.6, 2
        width = total_width / n

        plt.figure(figsize=(8, 8))

        for i in range(len(xxx)):
            xxx[i] = xxx[i] + width
        plt.bar(xxx, l, width=width, label='number of Instance', fc='mediumblue')
        # for a, b in zip(x, y):
        #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=11)
        plt.legend()
        plt.savefig(path_tmp + "/Histogram.pdf")
        plt.close()



# Main/entry method
if __name__ == '__main__':
    # Collate command line arguments
    parser = argparse.ArgumentParser(description='Parameters for visualising the embeddings via TSNE')

    parser.add_argument('--out_path', type=str, default='../output/embeddings_dataset/')
    parser.add_argument('--img_dir', type=str, default='../datasets/OpenSetCows2021/images/test')
    parser.add_argument('--embeddings_file', type=str, help='a folder name',
                        default='../output/embeddings_dataset')
    parser.add_argument('--perplexity', type=int, default=5,  # or 10 15        20 8
                        help="Perplexity parameter for t-SNE, consider values between 5 and 50")
    parser.add_argument('--components', type=int, default=8)  ### 45 20 182  8   186 will #167  171
    parser.add_argument('--png_only', type=int, default=1)
    parser.add_argument('--overlap_rate', type=float, default=0.3)

    parser.add_argument('--acc_folder', type=int, default=1) #
    parser.add_argument('--show_title', type=int, default=1)
    parser.add_argument('--save_txt', type=int, default=1)

    parser.add_argument('--lowRandNumber', type=int, default=1)
    parser.add_argument('--top2acc', type=int, default=1, help='exchange 2 clusters')

    parser.add_argument('--amplify', type=int, default=2, help='amplify size')
    parser.add_argument('--show_single_annotation', type=int, default=0, help='show images to user for labelling')
    parser.add_argument('--train_eb', type=int, default=200,
                        help='train embedding plot, 10 for small size, 200 for 4 days training')
    parser.add_argument('--topN', type=int, default=16, help='point based') # 4 8
    parser.add_argument('--gmm_max_iter', type=int, default=150)
    parser.add_argument('--plot_1by1', type=int, default=0)  # 180 gray
    parser.add_argument('--cal_recall', type=int, default=0)  # not important. test for 4 days     #val171 and fine tune
    # parser.add_argument('--date', default=[1, 1, 200, 200] , nargs='+', type=int,
    #                     help='start month, day, end ')  # 1, 1, 12, 30    2, 14, 2, 24    2,5,2,13  [1, 1, 200, 200]  [1, 200, 1, 200]
    parser.add_argument('--single_highlight', type=int, default=1) # gt as per id in the 2d position of all embeddings
    parser.add_argument('--db_iof', type=int, default= 0) # 1 db 0 iof. 2 no filter calculate all
    parser.add_argument('--seek_how_many_round', type=int, default=3) # always seed for 3 round. 4-5
    args = parser.parse_args()


    print("*******************   Loading embeddings   ************************")
    print(f"\nMax iter of GMM :{args.gmm_max_iter}", '\nNumber of GMM cluster:', args.components)
    print(f"perplexity :{args.perplexity}")
    print('mode:db_iof:',args.db_iof)
    # Get colour, reduction and original data, plot the tsne
    acc_test_knn = []
    ARI_test_knn = []
    acc_test_folder = []
    ARI_test_folder = []

    # date_folder = str(args.date[0]) + '-' + str(args.date[1]) + '_' + str(args.date[2]) + '-' + str(args.date[3])
    date_folder = get_date_now()
    modelinfo = "resnet50"
    tmp = args.embeddings_file
    args.embeddings_file = os.path.join(tmp, date_folder, modelinfo, '2combined/')
    file1 = sorted(os.listdir(args.embeddings_file)) # the test folder

    tmp = args.out_path
    for item in file1:
        if '.csv' not in item:
            for i in os.listdir(os.path.join(args.embeddings_file, item)):
                if i == 'folder_embeddings.npz':
                    print(f'\nOutput to folder: {item}')
                    args.out_path = os.path.join(tmp, date_folder, modelinfo, '3visual', item)
                    makedirs(args.out_path)
                    emd_path = os.path.join(args.embeddings_file, item, i)
                    path_tmp = os.path.join(args.out_path, os.path.basename(args.embeddings_file)[:-4])

                    # 2d after tsne, 128d data before tsne, corrected knn label, and colour, label from folder name, and colour
                    # load embedding and labels here
                    reduction, X_128, Tsne_LabelColor, Tsne_Label, GT_LabelColor, GT_Label, path, palette = plotEmbeddings_tsneMul(
                        args,
                        emd_path)  # both gt and tsne label

                    gmm = GaussianMixture(n_components=args.components, covariance_type='full', random_state=0,
                                          max_iter=args.gmm_max_iter)
                    # gmm_study(gmm, X_128, GT_Label, args), do not use it in the formal code
                    if args.single_highlight:
                        singleImag(reduction, GT_Label, args, palette, path, name='Single_gt') # defult=1 position 0 dots 1 images 2 image on the side
                        singleImag(reduction, GT_Label, args, palette, path, name='Single_g', position=2)


                    # outlier preparation
                    outlier_mask = (['a']) #['a'] means no filter from outlier methods
                    pred_LOF = (['a'])


                    print("..2d Tsne coloured with 128d Gmm")
                    PreLabelGMM, top_2_exchange_label, col, label_topN, RankLikely = PlotGMM(gmm,
                                                                                                                X_128,
                                                                                                                reduction,
                                                                                                                args,
                                                                                                                path,
                                                                                                                ax=plt)
                    #OUTPUT C_GMM_SELECT_D HERE
                    scatter_density(reduction, X_128, GT_Label, path_tmp, col, label_topN, path, args, outlier_mask, calculate_d=1,
                                    palette=palette, name='g_highlight', textupp=4, imgSize=30, pred_LOF=pred_LOF)
                    ###ACC AND DIFFERENCE between the label from will and GMM on a tsne plot     # # ACC AND DIFFERENCE
                    if args.train_eb < 20:
                        print('\nTrain_mode, do not calculate acc, Done')
                        # sys.exit(0)

                    if args.acc_folder:
                        # col, label_top8, top_2_highlight, args, name = '')
                        feb_pre_model, acc2, ARI2 = acc_plotall(GT_Label, GT_LabelColor, PreLabelGMM, col, label_topN,
                                                                top_2_exchange_label, RankLikely, args, path,
                                                                name='GMM-folder-Label on tsne')
                        if not 'best' in item:
                            acc_test_folder.append(acc2)
                            ARI_test_folder.append(ARI2)

                    # ****************************" Save "*********************************#
                    np.savez(os.path.join(tmp, f"acc.npz"), acc_test_knn=acc_test_knn, ARI_test_knn=ARI_test_knn,
                             acc_test_folder=acc_test_folder, ARI_test_folder=ARI_test_folder)

                    if args.save_txt:
                        np.savez(os.path.join(args.out_path, f"ings.npz"), embeddings=X_128, reduction=reduction,
                                 labels_folder=GT_Label, GMM_label=feb_pre_model)
