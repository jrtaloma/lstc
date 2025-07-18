import numpy as np
from sklearn import metrics
from scipy.special import comb


def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def evaluation(prediction, label):
    ri = rand_index_score(label, prediction)
    ari = metrics.adjusted_rand_score(label, prediction)
    nmi = metrics.normalized_mutual_info_score(label, prediction)
    return ri, ari, nmi
