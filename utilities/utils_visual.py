# Core libraries
import os,math, random
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

# Local libraries
from utils_visual_acc_roud import FristRd, SecondRd, printtopN


# toplabels, topNbyPoints, makedirs, count_if_enough, del_first, ACC,  draw_ellipse
# about k means: expand, getEuclidean, k_means, plot_cluster_no_label,

def toplabels(top_label = 0, t_p = 0, n = 8):
    count = 0
    while count < n:
        for i, item in enumerate(top_label):
            t_p[i][item] = float('-inf')
        top_next_label = np.argmax(t_p, axis=1)
        top_label = np.column_stack((top_label, top_next_label))
        count += 1
    return top_label

def topNbyPoints(label_input, pre_label_gmm_feedback, label_topN, All_pair, n, cal_recall = 0):

    '''
    ACC AND RECALL
    '''

    # Accuracy
    colle_acc = []
    cluster = []
    plot_accum = []
    count = 1
    Top_l = pre_label_gmm_feedback
    while count < n:
        for m2label in label_topN[:, count]:
            idx = np.where(m2label == All_pair[:, 0])
            if len(idx[0]) > 0:
                assert len(idx[0]) <= 1
                cluster.append(int(All_pair[idx, 1]))
            else:   # if did not find labels
                cluster.append(-2)
        Top_l = np.vstack((Top_l, cluster))
        count +=1
        cluster = []

        # calculate top N accuracy
        plot_single = []
        c_acc = 0
        for i in range(len(label_input)):
            gt = int(label_input[i])
            if gt in Top_l[:, i]: #find a label in top n
                c_acc += 1
                plot_single.append(0)
            else:                   #did not find. mark as 1
                plot_single.append(1)
        colle_acc.append(round(c_acc / len(label_input) * 100, 2))
        plot_accum.append(plot_single)

    # calculate top N recall,f1,precision
    result_prf = []
    if cal_recall:
        del_ = 0
        result_prf = np.zeros((n,3))
        for j in range(0,n):
            weighted_prf = []
            for i in np.unique(label_input):
                label_mask = (label_input == i)  # for gt
                label_indices = np.where(label_mask)[0]
                candidate = Top_l[:j+1, label_indices]
                TP = 0
                FP = 0
                gt = len(label_indices)
                for k in range(gt):
                    if i in candidate[:, k]:
                        TP += 1
                FN = gt - TP
                label_mask = 1 - label_mask # the rest
                label_indices = np.where(label_mask)[0]
                expt_gt = len(label_indices)
                candidate_expt_gt = Top_l[:j + 1, label_indices]
                for k in range(expt_gt):
                    if i in candidate_expt_gt[:, k]:
                        FP += 1
                ALL = len(label_input)
                #TN = ALL-FN-FP-TP

                if FP == 0 and TP==0:
                    recall = 0
                else:
                    recall = TP / (TP + FN)

                if TP == 0 and FP==0:
                    precision = 0
                else:
                    precision = TP / (TP + FP)

                if precision == 0 and recall==0:
                    F1 = 0
                else:
                    F1 = 2 * precision * recall / (precision + recall)

                weight = gt/ALL
                del_ += weight
                weighted_prf.append([precision*weight,recall*weight,F1*weight])
            weighted_prf = np.array(weighted_prf)
            result_prf[j] = np.round(np.sum(weighted_prf,axis=0),2)

    return colle_acc, np.array(plot_accum), result_prf


def makedirs(path):  # another format is in dncnn
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def count_if_enough(top_2_highlight, args):
    #count if there is enough spots with same high likely hodd labels ----log in save_calculated
    b = []
    save_calculated = []
    pb = []
    for i, item in enumerate(top_2_highlight):
        b.append(item[0] + item[1])
        tm0 = item[0]
        tm1 = item[1]
        count = 0
        if item not in save_calculated:
            for k, t in enumerate(top_2_highlight):
                if tm0 == t[1]:
                    if tm1 == t[0]:
                        count += 1
                if item == t:
                    count += 1
            if count >= args.lowRandNumber:
                save_calculated.append(item)
                save_calculated.append([item[1], item[0]])
                pb.append(count / len(top_2_highlight))
                pb.append(count / len(top_2_highlight))
    return save_calculated


def accBasedCluster(top_2_exchange_label, All_pair, label_input, pre_label_gmm_feedback):
    lover_idex = []
    SaveEnough = top_2_exchange_label
    first_SE = []  # get initial number for convinence.
    for tmtmt in SaveEnough:
        first_SE.append(tmtmt[0])

    f_tmp = []
    tmpAppend = []
    for item in All_pair:
        gt_l = item[1]
        gmm_l = item[0]
        if int(gmm_l) in first_SE:
            for i, pp in enumerate(SaveEnough):
                if pp[0] == gmm_l:
                    if len(SaveEnough[i]) == 2:
                        SaveEnough[i] = [pp[0], pp[1], gt_l]  # GMM ORIGINAL, SECOND CONFIDENT, GT OF FIRST
                        f_tmp.append(gt_l)
                    else:
                        tmpAppend.append([pp[0], pp[1], gt_l])
                        f_tmp.append(gt_l)
    save_enough = SaveEnough + tmpAppend  # now if 3 in []: top1 original gmm top2 original gmm, gmm cluster label of top1

    collect_exchange = []
    for i, pp in enumerate(save_enough):
        if len(save_enough[i]) == 3:
            collect_exchange.append(save_enough[i])

    for i, pp in enumerate(collect_exchange):
        SECOND_gmm = pp[1]
        for item in All_pair:
            gmm_l = item[0]
            gt_l = item[1]
            if SECOND_gmm == gmm_l:
                collect_exchange[i] = [*pp, gt_l]  # pp[0], pp[1], gt_l]  # GMM ORIGINAL, SECOND CONFIDENT, GT OF FIRST
                # f_tmp.append(gt_l)

    ####accuracy
    c_acc = 0
    count_black = 0
    tmppp = []
    for i in collect_exchange:  # find the reason why length is 3 (because some did not find label)
        if len(i) == 4:
            tmppp.append(i)
    collect_exchange = tmppp

    for i in range(len(label_input)):
        A_gt = label_input[i]
        B_cluster = pre_label_gmm_feedback[i]

        if A_gt == B_cluster:
            c_acc += 1
        elif B_cluster in f_tmp:  # Bbb in f_tmp:   # judge if Bbb is the sesond choice
            for item in collect_exchange:
                if B_cluster == item[2] and A_gt == item[3]:
                    c_acc += 1
                    lover_idex.append(i)
        if B_cluster < 0:  # double check
            count_black += 1
    AccLover = c_acc / len(label_input) * 100
    blank2 = count_black / len(label_input) * 100

    assert(len(set(lover_idex)) == len(lover_idex)) # KNN can not pass
    print(f'Accuracy exchange 2clusters: {round(AccLover, 3)}   Blank_rate: {round(blank2, 2)}% ')
    return lover_idex


def ACC(label_input, pre_label_gmm, top_2_exchange_label, label_topN, args):  # cluster and vote
    #global class_gmm, not_class_gt, not_class_gmm,  class_gt, luckboy, First_unity, count_minus

    ######################### assign the gmm clusters labels  with 3 round ##############################


    class_gt = set(label_input)
    class_gmm = set(pre_label_gmm)
    count_minus = -1
    # first rd
    blank_rate1, accFrist, pre_label_gmm_feedback, First_unity, luckboy = FristRd(label_input, pre_label_gmm, class_gt, class_gmm, count_minus, args )

    pre_label_gmm_feedback, not_class_gt, not_class_gmm, second_unity = SecondRd(pre_label_gmm_feedback, class_gt, class_gmm, luckboy, First_unity, count_minus, label_input, pre_label_gmm)

    diff_mask = (label_input != pre_label_gmm_feedback)
    difference_label = np.where(diff_mask)[0]

    #####print, blank rate acc ##############
    label_mask = (np.array(pre_label_gmm_feedback) <= -1)
    label_indices = np.where(label_mask)[0]
    blank_rate2 = (len(label_indices) / len(pre_label_gmm_feedback))*100
    print(f'\nTop1 results after 2 rds:', end ='          ')
    print('Did not calculate Precision, Recal and F1')
    #precision_we = precision_score(label_input, pre_label_gmm_feedback, average='weighted') # average='macro' micro
    #f1_we = f1_score(label_input, pre_label_gmm_feedback, average='weighted')
    #rc_we = recall_score(label_input, pre_label_gmm_feedback, average='weighted')
    #print(f'Recal: {round(rc_we,2)}    Precision:{round(precision_we,2)},   F1:{round(f1_we,2)}')
    print(f'Accuracy gmm cluster 1st rd: {round(accFrist, 3)}   Blank_rate: {round(blank_rate1, 2)}%   overlap_th:{args.overlap_rate} ')
    acc2nd = sum(1 for a, b in zip(label_input, pre_label_gmm_feedback) if a == b) / len(label_input) * 100
    print(f'Accuracy gmm cluster 2nd rd: {round(acc2nd,3)}   Blank_rate: {round(blank_rate2, 2)}%   overlap_th:{0}' )
    # print(f'2rd Acc:{round(accuracy_score(label_input, pre_label_gmm_feedback)*100,3)}') the same as acc2nd
    #####print, blank rate acc end ###########


    # merge all trustble matched paris (1st 2nd)
    if len(second_unity) == 0:
        All_pair = First_unity
    else:
        All_pair = np.concatenate((First_unity, second_unity), axis=0)

    ########### 3rd round randomly assigned for unclustered (clusters) #################
    Third_unity = []
    if len(not_class_gt) > 0:
        print('**In the 3rd rd match gmm to gt with 0 overlap in the second rd')
        for x, y in zip(not_class_gmm, not_class_gt):
            Third_unity.append([x,y,0])
        All_pair = np.concatenate((All_pair, Third_unity), axis=0)

    ######################### assign the gmm clusters labels  with 3 round end##############################

    if args.topN: # TOP N ACC BASEED ON points #####
        TopN_Acc, top_4animal, result_prf = topNbyPoints(label_input, pre_label_gmm_feedback, label_topN, All_pair, args.topN, args.cal_recall)
        printtopN(TopN_Acc, acc2nd, result_prf)

    ######################### TOP 2 exchage highlight ACC ############################
    AccLover = 0
    lover_idex = 0
    if args.top2acc:
        lover_idex = accBasedCluster(top_2_exchange_label, All_pair, label_input, pre_label_gmm_feedback)
                                                                                          # Lover 2 works when --top2acc=1
    return np.array(pre_label_gmm_feedback), np.array(difference_label), round(acc2nd, 2), np.array(lover_idex), round(AccLover, 2), top_4animal


def expand(x, y, gap=1e-4):
    add = np.tile([0, gap, np.nan], len(x))
    x1 = np.repeat(x, 3) + add
    y1 = np.repeat(y, 3) + add
    return x1, y1
