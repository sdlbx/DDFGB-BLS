# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_pred, y_score):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    pos = (y_true == 1)
    neg = (y_true == -1)
    p = np.sum(pos)
    n = np.sum(neg)

    tp = np.sum(pos & (y_pred == 1))
    tn = np.sum(neg & (y_pred == -1))

    tpr = tp / p if p != 0 else 0.0
    tnr = tn / n if n != 0 else 0.0
    gmean = 100.0 * np.sqrt(tpr * tnr)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5

    return auc, gmean