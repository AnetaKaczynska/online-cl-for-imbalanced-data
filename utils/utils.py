import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def display_results(table, n_classes=10):
    columns = ['train set']
    for c in range(n_classes):
        columns.append(c)
    columns.append('test acc')
    df = pd.DataFrame(table, columns=columns)
    df.set_index('train set')
    return df


def val_acc_per_subset(y_true, y_predicted):
    y_true = np.asarray(y_true)
    y_predicted = np.asarray(y_predicted)
    val_acc = []
    n_classes = np.max(y_true) + 1
    for i in range(10):
        idx = np.where(y_true == i)
        val_acc.append(accuracy_score(y_true[idx], y_predicted[idx]))
    return val_acc