import numpy as np
from sklearn.metrics import (precision_recall_curve, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, cohen_kappa_score)

# accuracy
def Accuracy(y_actual, y_pred): # Two arrays: actual label and prediction
    return (np.sum(y_actual == y_pred)/y_actual.shape[0])*100

# Balanced Accuracy
def BalancedAccuracy(y_actual, y_pred):
    return balanced_accuracy_score(y_actual, y_pred)*100

def Precision(y_actual, y_pred):
    return precision_score(y_actual, y_pred, average='macro')*100

def F1(y_actual, y_pred):
    return f1_score(y_actual, y_pred, average='macro')*100

def Kappa(y_actual, y_pred):
    return cohen_kappa_score(y_actual, y_pred)*100

def Multiclass_gmean(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    geometric_mean = np.prod(recalls) ** (1 / len(recalls)) *100
    return geometric_mean


def get_all_metrics(Y_true, Y_pred, time_taken):
    accuracy = Accuracy(Y_true, Y_pred)
    balanced_accuracy = BalancedAccuracy(Y_true, Y_pred)
    precision = Precision(Y_true, Y_pred)
    f1_score = F1(Y_true, Y_pred)
    multiclass_gmean = Multiclass_gmean(Y_true, Y_pred)
    kappa = Kappa(Y_true, Y_pred)

    return {"Kappa"         : kappa,
            "Accuracy"      : accuracy,
            "bAccuracy"     : balanced_accuracy,
            "Precision"     : precision,
            "F1"            : f1_score,
            "mGmean"        : multiclass_gmean,
            "Time"          : time_taken}