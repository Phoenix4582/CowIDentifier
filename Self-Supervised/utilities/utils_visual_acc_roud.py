# Core libraries
import os,math, random
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import collections
import seaborn as sns
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score

# Local libraries

# FristRd, SecondRd, del_first, printtopN

def del_first(First_unity):

    duplicate1 = [item for item, count in collections.Counter(First_unity[:,0]).items() if count > 1]
    log = []
    for i in duplicate1:
        for j, item in enumerate(First_unity):
            if i == item[0]:
                log.append(j)
                break
    pp = tuple(log)
    npop_remove =np.delete(First_unity, pp, axis=0) # remove the first meet
    duplicate2 = [item for item, count in collections.Counter(npop_remove[:, 0]).items() if count > 1]
    return npop_remove, duplicate2  # the duplicated after remove every first.

def FristRd(label_input, pre_label_gmm, class_gt, class_gmm, count_minus, args):
    normal_p = []
    luckboy = []

    pre_label_gmm_feedback = [count_minus] * len(label_input)

    print ('\nClasses of GT labels: {}'.format (len(class_gt)))

    for label in class_gt:
        married_overlap = []
        married_pair = []
        label_mask = (label_input == label)
        label_indices = np.where(label_mask)[0]
        if len(label_indices) < 2:
            print('Only 2 images in ID {}'.format(label_indices))

        for lab_pre in class_gmm:
            label_mask_pre = (pre_label_gmm == lab_pre)
            label_indices_pre = np.where(label_mask_pre)[0]
            c = list(set(label_indices_pre).intersection(set(label_indices)))
            overlap_rate = len(c) / len(label_indices)
            # find gmm have high overlap(0.3) first in gt
            if overlap_rate > args.overlap_rate: #as we have second rd.
                married_overlap.append(overlap_rate)
                married_pair.append([lab_pre, label, overlap_rate]) #allocate gt(second) to pre gmm (first)
            elif overlap_rate > 0:
                luckboy.append(label)
        if len(married_overlap) > 0:
            indexBest = married_overlap.index(max(married_overlap))
            normal_p.append(married_pair[indexBest])

    First_unity = sorted(normal_p, key=lambda a_entry: a_entry[2])  # ovelap low to high
    First_unity = np.array(First_unity)
    First_unity, clean_mark = del_first(First_unity)   #removed first meet
    while (len(clean_mark) != 0):
        First_unity, clean_mark = del_first(First_unity)
        print('1$', end = '')

    for itm in First_unity:
        label_mask_pre = (pre_label_gmm == itm[0])
        label_indices_pre = np.where(label_mask_pre)[0]
        for item in label_indices_pre:
            pre_label_gmm_feedback[item] = itm[1]

    label_mask = (np.array(pre_label_gmm_feedback) <= count_minus)
    label_indices = np.where(label_mask)[0]
    blank_rate1 = (len(label_indices) / len(pre_label_gmm_feedback))*100
    accFrist = sum(1 for a, b in zip(label_input, np.array(pre_label_gmm_feedback)) if a == b) / len(label_input) * 100

    return blank_rate1, accFrist, pre_label_gmm_feedback, First_unity , luckboy

def SecondRd(pre_label_gmm_feedback, class_gt, class_gmm, luckboy, First_unity, count_minus, label_input, pre_label_gmm):
    ########### second round find gt label in prediction 被高分挤掉了.重新找合适的.
    # find labels that did not assinged in gmm and the labels not appear in gt
    second_unity = []
    not_class_gt=[]
    not_class_gmm = []
    if min(pre_label_gmm_feedback) <= count_minus:
        tmp_gmm_asiign = np.array(First_unity)[:, 1]
        not_class_gt =  (class_gt - set(tmp_gmm_asiign)).intersection(set(luckboy))
        not_class_gmm = class_gmm - set(np.array(First_unity)[:, 0])
        print('\n**In the 1st rd, did not find {} GMM label: {} \nand {} gt label {}'.format(len(not_class_gmm),not_class_gmm, len(not_class_gt), not_class_gt))
        toBmarried_overlap = []
        toBmarried = []
        normal_pS = []
        second_unity = []

        for label_a in (not_class_gt):
            label_mask = (np.array(label_input) == label_a)
            label_indices = np.where(label_mask)[0]
            for label_b in not_class_gmm:
                label_mask_pre = (np.array(pre_label_gmm) == label_b)
                label_indices_pre = np.where(label_mask_pre)[0]
                c = list(set(label_indices_pre).intersection(set(label_indices)))
                overlap_rate = len(c) / len(label_indices)
                # find gmm have high overlap(10) first in gt
                if overlap_rate > 0:
                    toBmarried_overlap.append(overlap_rate)
                    toBmarried.append([label_b, label_a, overlap_rate]) #allocate gt(second) to pre gmm (first)
            if len(toBmarried_overlap) > 0:
                indexBest = toBmarried_overlap.index(max(toBmarried_overlap))
                normal_pS.append(toBmarried[indexBest])

        if len(normal_pS) != 0:
            # ovelap low to high. make sure most can cover less in the next part
            normal_pS = sorted(normal_pS, key=lambda a_entry: a_entry[2])
            second_unity = np.array(normal_pS)
            second_unity, clean_mark = del_first(second_unity)  # removed first meet
            while (len(clean_mark) != 0):
                second_unity, clean_mark = del_first(second_unity)
                print('2$',end =' ')
            print('\n')
            for itm in second_unity:
                label_mask_pre = (pre_label_gmm == itm[0])
                label_indices_pre = np.where(label_mask_pre)[0]
                for item in label_indices_pre:
                    pre_label_gmm_feedback[item] = itm[1]

    print('**In the 2nd rd,',end =' ')
    if min(pre_label_gmm_feedback) <= count_minus:
        if len(normal_pS) == 0:
            print('same as 1st rd')
        else:
            tmp_gmm_asiign = np.array(second_unity)[:, 1]
            not_class_gt = not_class_gt - set(tmp_gmm_asiign)
            not_class_gmm = not_class_gmm - set(np.array(second_unity)[:, 0])
            print('did not find {} GMM label: {} \nand {} gt label {}'.format(len(not_class_gmm),
                                                                                             not_class_gmm,
                                                                                             len(not_class_gt),
                                                                                             not_class_gt))
    else:
        print("Matched all")
    ########### second round find gt label in prediction end
    return pre_label_gmm_feedback, not_class_gt, not_class_gmm, second_unity



def printtopN(TopN_Acc, acc2nd, result_prf):
    print(f'Top N Accuracy based on cluster then points')
    print(
        '\n------------------------------------------------------------------------------------------------------------------')

    print(f' N   ', end='')
    print(f'1', end='      ')
    for i in range(len(TopN_Acc)):
        print(f'{str(i + 2).ljust(7)}', end='')
    print('\n', end='')

    print(f'ACC  ', end='')
    print(f'{str(round(acc2nd, 2)).ljust(7)}', end='')
    for i in range(len(TopN_Acc)):
        print(f'{TopN_Acc[i]}', end='  ')

    # print('\n', end='')
    # print(f'Rec  ', end='')
    # for i in range(len(result_prf)):
    #     print(f'{str(result_prf[i][1]).ljust(7)}', end='')
    #
    # print('\n', end='')
    # print(f'PRE  ', end='')
    # for i in range(len(result_prf)):
    #     print(f'{str(result_prf[i][0]).ljust(7)}', end='')
    #
    # print('\n', end='')
    # print(f'F1   ', end='')
    # for i in range(len(result_prf)):
    #     print(f'{str(result_prf[i][2]).ljust(7)}', end='')
    print(
        '\n------------------------------------------------------------------------------------------------------------------')
