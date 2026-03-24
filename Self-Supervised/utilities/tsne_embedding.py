# Core libraries
import os
import cv2, math, itertools
import numpy as np
import pandas as pd
import scipy.stats

from collections import Counter

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# expand, imscatter, scatter_overlap, scatter_images, scatter_singel(need time ), scatter_density

def expand(x, y, gap=1e-4):
    '''
    for overlap
    '''
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    return x1, y1


def imscatter(x, y, images, args, ax=None, zoom=1, w=10):
    '''
    plot images in the position of embeddings
    '''
    if ax is None:
        ax = plt.gca()
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, image in zip(x, y, images):
        # if os.path.basename(image)[12]!= '0':
        #     pass
            #print('Found an image with augmented angle')
            #continue
        im = cv2.imread(os.path.join(args.img_dir, image))
        im = cv2.resize(im, (w, int(w/2)))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_f = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im_f, (x0, y0), xycoords='data', frameon=False)
        #ik=ax.add_artist(ab)
        artists.append([ax.add_artist(ab)])
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def scatter_images(x, path, args, img_path=None, number = 1):
    fig, bx = plt.subplots()
    bx.axis('off')
    bx.axis('tight')
    plt.tight_layout()
    imscatter(x[:, 0], x[:, 1], images=img_path, args= args, ax=bx, zoom=1, w=10)
    plt.savefig(path + "scatter_image1.pdf")

    if number > 1:
        imscatter(x[:, 0], x[:, 1], images=img_path, args= args, ax=bx, zoom=1, w=20)
        plt.savefig(path + "scatter_image2.png")
    plt.close()





def scatter_overlap(x, labels, filename, palette=None ,show_class=0, name=''):
    # Choose a color palette with seaborn Randomly shuffle with the same seed
    num_classes = np.unique(labels).shape[0]

    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    # Convert labels to int
    labels = labels.astype(int)
    # reorder lists

    x_sort = (x[np.argsort(labels)])
    #l =labels.tolist()
    #m=x.tolist()
    #l, m = (zip(*sorted(zip(l, m))))
    # Map the colours to different labels

    labels_sort = np.sort(labels)

    last_label = 0
    last_i = 0
    plt.rcParams['lines.solid_capstyle'] = 'round'
    for i in range(len(labels_sort)):
        label_colours = np.array([palette[labels_sort[i-1]]])
        if labels_sort[i] != last_label:
            ax.plot(*expand(x_sort[:, 0][last_i:i], x_sort[:, 1][last_i:i]), lw=7, c=label_colours[0],
                    alpha=0.5)
            last_i = i
        last_label = labels_sort[i]

    if show_class:
        for i in range(num_classes):
            # Position of each label.
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            if math.isnan(xtext):
                continue
            txt = ax.text(xtext, ytext, str(i), fontsize=11)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=3, foreground="w"),
                PathEffects.Normal()])

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.title('{} Images Graph'.format(str(len(labels))), fontsize='large', fontweight = 'bold')# 设置字体大小与格式
    plt.tight_layout()

    #plt.suptitle('s')
    print(f"Saved visualisation")
    plt.savefig(filename +'B_overlap_'+ name+ ".png")
    plt.close()


def scatter_singel(x, labels, filename, args, palette=None, img_path='', position = 0):
    '''
    scatter label by label
    '''

    # Create our figure/plot
    num_classes = np.unique(labels).shape[0]
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    # Convert labels to int
    labels = labels.astype(int)
    # Map the colours to different labels
    label_colours = np.array([palette[labels[i]] for i in range(labels.shape[0])])

    mean = []
    std  = []
    x_count=[]
    l = []
    for count in range(min(labels),max(labels)+1):
        if count in labels:
            # if count > 2:
            #     break
            # Which labels should we highlight (the "difficult" individuals)
            highlight_labels = [count]    #highlight_labels = [54, 69, 73, 173]
            # Colour for non-highlighted points
            label_colours = np.zeros(label_colours.shape)
            # Plot all the points
            plt.rcParams['lines.solid_capstyle'] = 'round'
            ax.plot(*expand(x[:, 0], x[:, 1]), lw=7, c=label_colours[0], alpha=0.1)

            # Highlighted points
            h_pts = np.array([x[i, :] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
            h_colours = np.array([palette[labels[i]] for i in range(labels.shape[0]) if labels[i] in highlight_labels])
            h_images = np.array([img_path[i] for i in range(labels.shape[0]) if labels[i] in highlight_labels])

            mean.append(np.mean(h_pts[:, 0]).tolist())
            l.append(len(h_pts))
            std.append(int((np.var(h_pts[:, 0])+np.var(h_pts[:, 1])).tolist()))
            x_count.append(count)

            if position ==1:
                imscatter(h_pts[:, 0], h_pts[:, 1], images=h_images, args = args, ax=ax, zoom=1, w=30)
            elif position == 0:
                ax.scatter(h_pts[:, 0], h_pts[:, 1], lw=0, s=10, c=h_colours, marker="o", alpha=0.7)
            else:
                ax.scatter(h_pts[:, 0], h_pts[:, 1], lw=0, s=10, c=h_colours, marker="o", alpha=0.7)
                h_range= max(x[:, 1])-min(x[:, 1])
                turn = min(x[:, 1])
                h_pts[:, 1] = np.array([max(x[:, 1]) - i*7 for i in range(len(h_pts))])
                h_pts[:, 0] = np.array([min(x[:, 0]) - 15 for i in range(len(h_pts[:, 1]))])

                qqq = (h_pts[:, 1][0] - h_pts[:, 1][-1])/h_range
                searched = 0
                j = 0
                colum_plus = 1
                if qqq > 1:
                    for i in range (int(len(h_pts[:, 1])/(qqq+1)),len(h_pts[:, 1])):
                        if searched == 0:
                            if h_pts[:, 1][i] < turn:
                                number_per_col = i + 1
                                searched = 1
                        else:
                            h_pts[:, 1][i] = h_pts[:, 1][i] + max(x[:, 1])-min(x[:, 1])
                            h_pts[:, 1][i] = h_pts[:, 1][i-number_per_col]
                            h_pts[:, 0][i] = h_pts[:, 0][i-number_per_col] - 20

                            j +=1
                            if j >= number_per_col:
                                colum_plus += 1
                                j=0

                imscatter(h_pts[:, 0], h_pts[:, 1], images=h_images, args = args, ax=ax, zoom=1, w=40)

            ax.axis('on')
            ax.axis('tight')
            plt.tight_layout()
            plt.title('{} images for ID {} of {} classes'.format(str(len(h_pts)), count, num_classes), fontsize='large', fontweight='bold')  # 设置字体大小与格式
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            plt.savefig(filename + str(count) +"_Highlight.png", dpi=500)
            plt.cla()
    plt.close()

    return x_count, std, l


def distance_inter(x1, x2):
    Dist_sum = 0
    count = 0
    for item in list(itertools.product(x1, x2)): #get 1 from x1 and another form x2 to calculate distance
        m = (item[0]-item[1])** 2
        distance_one = np.sum(m)
        Dist_sum += distance_one
        count +=1
    if count !=0:
        Dist_ave = round(Dist_sum/count,1)
    else:
        Dist_ave = 100
    return Dist_ave


def find_(count_allquestions, c_self, whichLabel, label_topN, labels, textup, bx, x, color_se, args, path, xx, yy, images2, imgSize, amplify, anchorFoderName, ahchorGtLable,
          X_128, last_i, i, last_path):
    cou_color = 0
    tmppaira = []
    tmppair2 = []
    acc_pari = []
    skipImg = 0
    for iii in whichLabel.keys():

        ClusterGroup = np.where(label_topN[:, 0] == iii)[0]

        # chose a median from this group to show the image
        labelObject = labels[int(np.median(ClusterGroup))] # real label
        tmpcandiFolder = path[int(np.median(ClusterGroup))][:5]
        label_every = []
        for item in ClusterGroup:
            label_every.append(labels[int(item)])
        maxlabel = max(label_every, key=label_every.count) # most labels in the ClusterGroup
        if maxlabel != labelObject:
            #print('labelObject change from', labelObject,'to', maxlabel)
            labelObject = maxlabel # path do not need to change
        ### collect gt, not useful
        # if labelObject == int(ahchorGtLable): #gt collection, seem it is not useful
        #     # 假设人工判断两个cluster 是一个label, 收集起来. else: we only show them to the annotator.
        #     tmppaira.append(anchorFoderName) #
        #     candidateFolder = path[int(np.median(ClusterGroup))][:5]
        #     if candidateFolder in tmppair2 or candidateFolder in tmppaira:
        #         skipImg = 1
        #     else:
        #         tmppair2.append(candidateFolder)

        # show the image:
        if args.show_single_annotation:
            bx.scatter(x[ClusterGroup, 0], x[ClusterGroup, 1], lw=0, s=40, c=color_se[cou_color % 3], marker="o",
                       alpha=0.8)
            cou_color += 1

            if not skipImg:
                xtext, ytext = np.median(x[ClusterGroup, :], axis=0)
                txt = bx.text(xtext, ytext + int(textup*amplify/1.2), str(labelObject)+'  '+tmpcandiFolder, fontsize=int(11*amplify/1.2),c='k')
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=3, foreground="w"),
                    PathEffects.Normal()])
                xx.append(xtext)
                yy.append(ytext)
                images2.append(os.path.join(args.img_dir, path[int(np.median(ClusterGroup))]))
        skipImg = 0

        ite_diver = tmpcandiFolder
        if last_path == ite_diver:
            c_self += 1

        find_first = 1
        i_begin = 0
        i_end = 0
        for it, track in enumerate(path):
            if ite_diver in track:
                if find_first:
                    i_begin = it
                    find_first = 0
                else:
                    i_end = it
        i_end += 1
        Dist_average = distance_inter(X_128[last_i:i, :], X_128[i_begin:i_end, :])

        #dself_1 = distance_inter(X_128[last_i:i, :], np.flip(X_128[last_i:i, :], axis=0)) #interdistance claculation, did not help
        #dself_2 = distance_inter(X_128[i_begin:i_end, :], np.flip(X_128[i_begin:i_end, :], axis=0))

        # find min_ max_ of every distance , not useful
        # for every 5 images
        # gap = 5
        # ct_fd1 = i - last_i
        # t1 = int(ct_fd1/gap)
        # ct_fd2 = i_end - i_begin
        # t2 = int(ct_fd2/gap)
        # Every_distance = []
        # for circle in range(t1):
        #     current_p = last_i+circle*gap
        #     for circle2 in range(t2):
        #         current_2 = i_begin+circle2*gap
        #         Dist_1 = distance_inter(X_128[current_p:current_p+gap, :], X_128[current_2:current_2+gap, :])
        #         Every_distance.append(Dist_1)
        #acc_pari.append([Dist_average, last_path, ite_diver, labels[last_i], labels[i_begin], Every_distance])

        acc_pari.append([Dist_average, last_path, ite_diver, labels[last_i], labels[i_begin]])
        count_allquestions += 1

    if args.show_single_annotation:
        imscatter(xx, yy, images2, args=args, ax=bx, zoom=1, w=int(imgSize*amplify/1.2))

    return bx, labelObject, tmppaira, tmppair2, count_allquestions, c_self, acc_pari


def scatter_density(x, X_128, labels, filename, col, label_topN, path, args, outlier_mask,
                    calculate_d=0, palette=None, name='', textupp = 1, imgSize=30, pred_LOF=(['a'])):
    # Get the number of classes (number of unique labels)
    global labelObject
    num_classes = np.unique(labels).shape[0]
    count_annotator = 0

    # Convert labels to int
    labels = labels.astype(int)
    # Map the colours to different labels
    label_colours = np.array([palette[labels[i]] for i in range(labels.shape[0])])
    # Plot the points
    if len(x) > 2000:
        amplify = args.amplify
    else:
        amplify = 2
    Replacex = x * amplify
    textup = int(textupp * amplify/2)
    plt.figure(figsize=(8 * amplify, 8 * amplify))
    ax = plt.subplot(aspect='equal')
    ax.scatter(Replacex[:, 0], Replacex[:, 1], lw=0, s=40*amplify, c=label_colours, marker="o", alpha=0.5)

    # listanchor = []
    # listcandidate = []
    # priority = []
    # listLabel = []
    # D = []
    # add the labels for each digit.
    count_allquestions = 0
    c_self = 0
    every_cluster = []
    if 1: # fold and unfold
        xs = []
        ys = []
        images = []
        last_path = path[0][:5]
        last_i = 0

        pd_Dis = []
        pd_F1 = []
        pd_F2 = []
        pd_L1 = []
        pd_L2 = []
        m11 = []
        pr = []
        pr10 = []
        round1or2 = []

        xx = []
        yy = []
        images2 = []

        proformlast = 0
        count_dbscan = 0


        # obtain images of each cluster for plotting
        for i, item in enumerate(path):
            mark_dbscan = 0
            if item[:5] != last_path:
                # Position of each label.
                # chose filter mode 1 db 0 iof. 2 no filter calculate all
                if args.db_iof == 1:
                    if len(outlier_mask[0]) == 1:
                        mark_dbscan = 1
                    else:
                        for item_out in outlier_mask[0]:
                            if item_out >= i:
                                break
                            if item_out >= last_i and item_out < i:
                                mark_dbscan=1
                                count_dbscan +=1

                                #print(labels[last_i],'+', last_path)
                                break
                    if mark_dbscan != 1:
                        last_path = item[:5]
                        last_i = i
                        proformlast = 0

                if args.db_iof == 0 and len(pred_LOF) >= 1: # 14th decc> 1 # iof mode and value is not ['a']
                    count_outlier_lof = Counter(pred_LOF[last_i:i])[-1]
                    if count_outlier_lof > 1:
                        mark_dbscan = 1
                    else:
                        last_path = item[:5]
                        last_i = i
                        proformlast = 0

                if args.db_iof == 2:
                    mark_dbscan =1

                if mark_dbscan==1:
                    every_cluster.append([last_i,i,last_path, labels[last_i]])
                    xtext, ytext = np.median(Replacex[last_i:i, :], axis=0)
                    if math.isnan(xtext):
                        continue
                                                         # label                  folder
                    txt = ax.text(xtext, ytext + textup, str(labels[last_i])+'  '+str(last_path), fontsize=int(11*amplify/2), c='r')
                    txt.set_path_effects([
                        PathEffects.Stroke(linewidth=3, foreground="w"),
                        PathEffects.Normal()])
                    xs.append(xtext)
                    ys.append(ytext)
                    # images
                    images.append(os.path.join(args.img_dir, path[last_i]))


                    # Hilight single for human annotator to merge them
                    if len(col) > 1 and calculate_d == 1:

                        pri = col[last_i:i]
                        pri.sort(reverse=True)
                        if int(len(pri) / 10) == 0:
                            pri_last10 = sum(pri)/(len(pri))
                            pri_first10 = pri_last10
                        else:
                            pri_last10 = sum(pri[:int(len(pri) / 10)]) / int(len(pri) / 10)
                            pri_first10 =  sum(pri[-int(len(pri) / 10):])  / int(len(pri) / 10)

                        #pri = np.median(col[last_i:i]) #each point belowg to the first cluster - second. small is what we want
                        pri = pri_first10
                        ######### about anchor ##########
                        anchorFoderName = str(last_path)
                        ahchorGtLable = str(labels[last_i])

                        bx = None
                        if args.show_single_annotation:
                            plt.figure(figsize=(8 * amplify, 8 * amplify))
                            bx = plt.subplot(aspect='equal')
                            # plot the background points
                            bx.scatter(Replacex[:, 0], Replacex[:, 1], lw=0, s=40*amplify, c=label_colours, marker="o", alpha=0.2)

                            # show text labels
                            # it is the tracklet [last_i:i] not gmm
                            xtext, ytext = np.median(Replacex[last_i:i, :], axis=0)
                            if math.isnan(xtext):
                                continue

                            txt = bx.text(xtext, ytext + textup, ahchorGtLable +'  '+ anchorFoderName, fontsize=int(30 * amplify / 2), c='r')
                            txt.set_path_effects([
                                PathEffects.Stroke(linewidth=3, foreground="w"),
                                PathEffects.Normal()])
                            # show images
                            xx = []
                            yy = []
                            xx.append(xtext)
                            yy.append(ytext)
                            images2 = []
                            images2.append(os.path.join(args.img_dir, path[last_i]))
                        ######### about anchor end##########

                        ######### about candidateds ##########
                        #if the first label is more likely itself, so we start from the second similar cluster.
                        secondLabels = Counter(label_topN[last_i:i, 1]) # label and The number
                        color_se = ['k', 'b', 'g']


                        bx, labelObject, listanchortm, listcandidatetm, count_allquestions, c_self, acc_pari = find_(count_allquestions, c_self, secondLabels, label_topN, labels, textup, bx, Replacex, color_se, args, path, xx, yy,
                                   images2, imgSize, amplify, anchorFoderName, ahchorGtLable, X_128, last_i, i ,last_path)
                        #to show to the user cluster and labels
                        # first_round
                        for every in acc_pari:
                            pd_Dis.append(every[0])
                            pd_F1.append(every[1])
                            pd_F2.append(every[2])
                            pd_L1.append(every[3])
                            pd_L2.append(every[4])
                            pr.append(pri)
                            pr10.append(pri_last10)
                            round1or2.append('')
                            if every[3] == every[4]:
                                m11.append(0)
                            else:
                                m11.append(-1)
                            break

                        #always seek for second rd
                        #if len(listanchortm) < 1: #work on to filter it!
                            # assume we know the label of the first round.  or let human see the secondLabels,
                            # if do not match, see 2nd or 3rd.
                        thirdLabels = Counter(label_topN[last_i:i, 2])
                        bx, labelObject, listanchortm, listcandidatetm, count_allquestions, c_self, acc_pari= find_(count_allquestions,c_self, thirdLabels,label_topN,labels,textup, bx, Replacex ,color_se, args, path,xx,yy,images2,imgSize,amplify, anchorFoderName, ahchorGtLable,X_128,last_i,i,last_path)
                        for every in acc_pari:
                            pd_Dis.append(every[0])
                            pd_F1.append(every[1])
                            pd_F2.append(every[2])
                            pd_L1.append(every[3])
                            pd_L2.append(every[4])
                            pr.append(pri)
                            pr10.append(pri_last10)
                            round1or2.append('2')
                            if every[3] == every[4]:
                                m11.append(0)
                            else:
                                m11.append(-1)
                            break

                        #always seek for 3 rd rd
                        if args.seek_how_many_round >= 3:
                            nextLabels = Counter(label_topN[last_i:i, 3])
                            bx, labelObject, listanchortm, listcandidatetm, count_allquestions, c_self, acc_pari= find_(count_allquestions,c_self, nextLabels,label_topN,labels,textup, bx, Replacex ,color_se, args, path,xx,yy,images2,imgSize,amplify, anchorFoderName, ahchorGtLable,X_128,last_i,i,last_path)

                            for every in acc_pari:
                                pd_Dis.append(every[0])
                                pd_F1.append(every[1])
                                pd_F2.append(every[2])
                                pd_L1.append(every[3])
                                pd_L2.append(every[4])
                                pr.append(pri)
                                pr10.append(pri_last10)
                                round1or2.append('3')
                                if every[3] == every[4]:
                                    m11.append(0)
                                else:
                                    m11.append(-1)
                                break

                        if args.seek_how_many_round >= 4:
                            nextLabels = Counter(label_topN[last_i:i, 4])
                            bx, labelObject, listanchortm, listcandidatetm, count_allquestions, c_self, acc_pari= find_(count_allquestions,c_self, nextLabels,label_topN,labels,textup, bx, Replacex ,color_se, args, path,xx,yy,images2,imgSize,amplify, anchorFoderName, ahchorGtLable,X_128,last_i,i,last_path)

                            for every in acc_pari:
                                pd_Dis.append(every[0])
                                pd_F1.append(every[1])
                                pd_F2.append(every[2])
                                pd_L1.append(every[3])
                                pd_L2.append(every[4])
                                pr.append(pri)
                                pr10.append(pri_last10)
                                round1or2.append('4')
                                if every[3] == every[4]:
                                    m11.append(0)
                                else:
                                    m11.append(-1)
                                break


                        # for gt, C_annotator.csv
                        # not wroking well now ...
                        # if len(listanchortm) > 0 and len(listcandidatetm) > 0:
                        #     listLabel.append(ahchorGtLable)
                        #     listanchor.append(listanchortm[0])
                        #     #delete repete
                        #     listcandidate.append(list(dict.fromkeys(listcandidatetm)))
                        #     priority.append(pri)
                        #     tmtm = []
                        #     for ite_diver in listcandidatetm:
                        #         find_first = 1
                        #         for it, track in enumerate(path):
                        #             if ite_diver in track:
                        #                 if find_first:
                        #                     i_begin = it
                        #                     find_first = 0
                        #                 else:
                        #                     i_end = it
                        #         i_end += 1
                        #         Dist_average = distance_inter(X_128[last_i:i, :], X_128[i_begin:i_end,:])
                        #         #print(labels[last_i], last_path, ite_diver, Dist_average)
                        #         tmtm.append(Dist_average)
                        #     D.append(tmtm)
                        # count_annotator += 1

                        # show image and the candidate similar cluster for each cluster
                        if args.show_single_annotation:
                            bx.axis('tight')
                            plt.tight_layout()   # "%08d" % (pri) +
                            plt.savefig(filename + 'user/A_annotator_' + str(labels[last_i]) + '_' + str(last_i) + ".png")
                            plt.close()

                    last_path = item[:5]
                    last_i = i
                    proformlast = 1

        if len(col) > 1:
            print(f'GMM All questions: {count_allquestions} in {count_annotator} images, filter self did not merge : {count_allquestions-c_self}')
            # the last one
            if proformlast:
                xtext, ytext = np.median(Replacex[last_i:, :], axis=0)
                txt = ax.text(xtext, ytext + textup, str(labels[i]) + str(item[:5]), fontsize=11*int(amplify/2)) #?
                txt.set_path_effects([
                    PathEffects.Stroke(linewidth=3, foreground="w"),
                    PathEffects.Normal()])
                xs.append(xtext)
                ys.append(ytext)
                images.append(os.path.join(args.img_dir, path[i]))

        # images = []
        # for item in sorted(os.listdir(args.img_dir)):
        #     for image in os.listdir(os.path.join(args.img_dir,item)):
        #         if len(image) < 2: # 3 levels folder
        #             for image_in in os.listdir(os.path.join(args.img_dir, item,image)):
        #                 images.append(os.path.join(args.img_dir, item, image, image_in))
        #                 break
        #         else: # 3 levels folder
        #             images.append(os.path.join(args.img_dir,item,image))
        #             break

        imscatter(xs, ys, images, args=args, ax=ax, zoom=1, w=imgSize * int(amplify/2))

        # if len(listanchor) > 0 and calculate_d == 1:
        #     # delete repete tracklets data before saving
        #     data1 = pd.DataFrame({'listLabel':listLabel, 'listanchor': listanchor, 'listcandidate': listcandidate,'priority':priority, 'Dist_average':D} )
        #     print(f'save to', os.path.join(args.out_path,'C_annotator.csv'))
        #     data1.to_csv(os.path.join(args.out_path,'C_annotator.csv'))

        print(f'Number of clusters:', count_dbscan)

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.tight_layout()
    plt.savefig(filename + 'A_density_' + name + ".png")
    plt.close()

    # log gmm inter distances:
    if calculate_d == 1:
        data3 = pd.DataFrame(
        {'Distance': pd_Dis, 'folder1': pd_F1, 'folder2': pd_F2, 'label1': pd_L1, 'label2': pd_L2,
         'priority_m': pr, 'priority10': pr10,'round1or2': round1or2,'label_':m11 }) #'min': min_, 'max': max_}) ##,  'each distance': each_dist})
        print(f'save to', os.path.join(args.out_path, 'C_GMM_SELECT_DIS.csv'))
        data3.to_csv(os.path.join(args.out_path, 'C_GMM_SELECT_DIS.csv'))

    if calculate_d == 1 and 1==0:
        ## calculate inter distances:
        bufferinfo = []
        buffer2= []

        pd_Dis=[]
        pd_F1 =[]
        pd_F2 = []
        pd_L1 = []
        pd_L2 = []
        pd_L1_2 = []

        # intra distance
        pd_Dis_self = []
        # pd_F = []
        # pd_L = []

        count_eachcluster = 0
        nb = 4
        print(f'***Find the {nb-1} smallest distance per cluster in C_inter_distance.csv')
        #print(f'Saving C_intra_distance.csv')

        cycle = len(every_cluster)
        for item in list(itertools.product(every_cluster, every_cluster)):
            folder1 = item[0][2]
            folder2 = item[1][2]
            label_1 = item[0][3]
            label_2 = item[1][3]
            Dist_average = distance_inter(X_128[item[0][0]:item[0][1], :], X_128[item[1][0]:item[1][1], :])
            bufferinfo.append([folder1, folder2, label_1, label_2])
            buffer2.append(Dist_average)

            ## log intra distance
            # if folder1 == folder2:
            #     pass
                # pd_Dis_self.append(Dist_average)
                # pd_F.append(folder1)
                # pd_L.append(label_1)
                #print(count_eachcluster, Dist_average, folder1)  #for testing

            count_eachcluster += 1
            if count_eachcluster >= cycle:
                max_index2 = np.argsort(buffer2)[1:4] #smallest index 0 is it self, exclude it
                for pp in max_index2:
                    # print(buffer2[pp], bufferinfo[pp]) #for testing
                    pd_Dis.append(buffer2[pp])
                    pd_F1.append(bufferinfo[pp][0])
                    pd_F2.append(bufferinfo[pp][1])
                    pd_L1.append(bufferinfo[pp][2])
                    pd_L2.append(bufferinfo[pp][3])
                    if bufferinfo[pp][2]==bufferinfo[pp][3]:
                        pd_L1_2.append(0)
                    else:
                        pd_L1_2.append(-1)
                count_eachcluster = 0
                bufferinfo = []
                buffer2 = []


        #print(f'***The number of questions for inter-distance is {len(pd_Dis)}')
        data2 = pd.DataFrame(
            {'Distance': pd_Dis, 'folder1': pd_F1, 'folder2': pd_F2, 'label1': pd_L1, 'label2': pd_L2, 'label-': pd_L1_2})
        data2.to_csv(os.path.join(args.out_path, 'C_inter_distance.csv')) # 3 smallest for each cluster


        # data2 = pd.DataFrame({'Distance': pd_Dis_self, 'folder': pd_F, 'label': pd_L})
        # data2.to_csv(os.path.join(args.out_path, 'C_intra_distance.csv'))
