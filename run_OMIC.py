from utils.utils import seed_everything
from utils.metric_utils import get_all_metrics
from OMIC import NB_imb
from tqdm import tqdm
import numpy as np
import time
from utils.utils import get_ord_indices

def run_OMIC(X, Y, X_haphazard, mask, num_runs, params):
    X_masked = np.ones_like(X) * np.nan
    X_masked[mask.astype(bool)] = X[mask.astype(bool)]
    all_ord_indices = get_ord_indices(X_masked)

    # print('Preparing Data to be fed')
    data = []
    for i in range(0, len(Y)):
        X, x_mask, y = X_haphazard[i], mask[i], Y[i]
        X_dict = []
        d_dict = []
        for j in range(0, len(X)):
            if x_mask[j] == 1:
                X_dict.append(X[j])
            elif x_mask[j] == 0:
                X_dict.append(None)
        d_dict.append(X_dict)
        d_dict.append(float(y[0]))
        data.append(d_dict)

    eval_list = []
    dict_pred = {}

    for j in range(num_runs):
        seed_everything(j)
        Y_pred = []
        acc_all = []
        correct_cnt = 0
        num = 0

        start_time = time.time()

        c = []
        for k, g in data:
            c.append(g)
        model = NB_imb(len(data[0][0]), np.unique(Y), win_label= params['win_label'],
                       alpha=params['alpha'], max_w=params['max_w'], priors=True)
        for r, t in tqdm(zip(data, c)):
            num += 1
            X = r[0]
            y = r[1]
            y_pred = model.handleInstace(X, y, all_ord_indices)

            if y_pred == y:
                correct_cnt += 1
            acc_t = correct_cnt / num
            acc_all.append(acc_t)

            Y_pred.append(y_pred)

        taken_time = time.time() - start_time

        dict_pred[j] = Y_pred
        metrics = get_all_metrics(Y, np.array(Y_pred).reshape(-1, 1), taken_time)
        eval_list.append(metrics)
        del model

    return eval_list, dict_pred