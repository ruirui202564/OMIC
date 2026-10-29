from math import log, e, pi
import numpy as np
from collections import defaultdict
import math

class NB_imb(object):
    def __init__(self, D, all_classes, epsilon=np.inf, priors=False,
                 win_label=1000, sample_weight=True, drift=True,
                 alpha=15, min_w=1.0, max_w=50.0):
        self.alpha = alpha
        self.min_w = min_w
        self.max_w = max_w

        self.sample_weight = sample_weight
        self.drift = drift
        self.epsilon = epsilon
        self.D_ = np.int32(D)
        assert self.D_ > 0
        self.priors_ = priors

        self.instance_seen = 0
        self.all_classes = all_classes

        self.n_ = {}
        self.mean_ = {}
        self.M2_ = {}
        self.var_ = {}

        self.discrete_counts_ = {}  # {class: {feature_idx: {value: count}}}
        self.discrete_n_ = {}  # {class: numpy_array}

        self.win_label_size = win_label
        self.true_label_win = [-2] * self.win_label_size
        self.pos_window = 0
        self.window_counts = {}
        self.class_imb_ratio = {}

        self.total_ = 0.0
        self.class_counts_ = defaultdict(int)
        self.min_var_ = np.inf

        self._sorted_class_idx = sorted(
            range(len(self.all_classes)), key=lambda k: self.all_classes[k]
        )

    def _to_ord_set(self, ord_indices):
        if isinstance(ord_indices, (set, frozenset)):
            return ord_indices
        return frozenset(ord_indices)

    def _pdf_continuous(self, x, m, v):
        v += self.min_var_ / 1000 if self.min_var_ != np.inf else 1e-9
        return (-0.5 * log(2 * pi * v) - ((x - m) ** 2 / (2 * v)))

    def _prob_discrete(self, value, y, feature_idx):
        if y not in self.discrete_counts_ or feature_idx not in self.discrete_counts_[y]:
            return log(self.epsilon)

        value_count = self.discrete_counts_[y][feature_idx][value]
        total_count = self.discrete_n_[y][feature_idx]

        if total_count <= 0:
            return log(self.epsilon)

        unique_values = len(self.discrete_counts_[y][feature_idx])
        k = max(unique_values, 1)

        alpha = 1.0
        smoothed_prob = (value_count + alpha) / (total_count + alpha * k)

        return log(max(smoothed_prob, 1e-10))

    def _get_class_prior(self, y):
        if self.priors_ and self.total_ > 0 and self.class_counts_[y] > 0:
            return log(self.class_counts_[y] / self.total_)
        else:
            return 0.0

    def predict(self, x, ord_indices):
        assert len(x) == self.D_
        self.instance_seen += 1

        if self.instance_seen == 1:
            default_probs = np.zeros(len(self.all_classes))
            default_probs[0] = 1
            return self.all_classes[0], default_probs

        ord_set = self._to_ord_set(ord_indices)

        D_ = self.D_
        all_classes = self.all_classes
        mean_dict = self.mean_
        var_dict = self.var_
        n_dict = self.n_
        discrete_counts = self.discrete_counts_
        discrete_n = self.discrete_n_
        eps = self.epsilon
        min_var = self.min_var_
        get_class_prior = self._get_class_prior

        scores = {}

        for y in all_classes:
            score = get_class_prior(y)

            mean_y = mean_dict.get(y)
            var_y = var_dict.get(y) if mean_y is not None else None
            n_y = n_dict.get(y) if mean_y is not None else None
            disc_counts_y = discrete_counts.get(y)
            disc_n_y = discrete_n.get(y) if disc_counts_y is not None else None

            for i in range(D_):
                xi = x[i]
                if xi is None or np.isnan(xi):
                    continue

                if i in ord_set:
                    if disc_counts_y is None or i not in disc_counts_y:
                        score += log(eps)
                        continue

                    counts_i = disc_counts_y[i]
                    value_count = counts_i[xi]
                    total_count = disc_n_y[i]

                    if total_count <= 0:
                        score += log(eps)
                        continue

                    k = max(len(counts_i), 1)
                    smoothed_prob = (value_count + 1.0) / (total_count + k)
                    score += log(max(smoothed_prob, 1e-10))
                else:
                    if mean_y is not None and n_y[i] > 0:
                        m = mean_y[i]
                        v = var_y[i]
                        v += (min_var / 1000) if min_var != np.inf else 1e-9
                        score += -0.5 * log(2 * pi * v) - ((xi - m) ** 2 / (2 * v))
                    else:
                        score += log(eps)

            scores[y] = score

        if not scores:
            default_probs = np.zeros(len(self.all_classes))
            default_probs[0] = 1
            return 0, default_probs

        classes = self.all_classes
        score_values = np.array(list(scores.values()))
        sorted_idx = self._sorted_class_idx

        if not np.any(np.isfinite(score_values)):
            default_probs = np.ones(len(classes)) / len(classes)
            prob_list = [default_probs[k] for k in sorted_idx]
            return classes[0], prob_list

        max_score = np.max(score_values)
        if not np.isfinite(max_score):
            default_probs = np.ones(len(classes)) / len(classes)
            prob_list = [default_probs[k] for k in sorted_idx]
            return classes[0], prob_list

        score_diff = score_values - max_score
        score_diff[~np.isfinite(score_diff)] = -700

        exp_scores = np.zeros_like(score_values)
        valid_mask = score_diff > -700
        exp_scores[valid_mask] = np.exp(score_diff[valid_mask])

        total_exp = np.sum(exp_scores)
        if total_exp == 0:
            probabilities = np.ones_like(score_values) / len(score_values)
        else:
            probabilities = exp_scores / total_exp

        prob_list = [probabilities[k] for k in sorted_idx]

        best_class = classes[np.argmax(probabilities)]

        return best_class, prob_list

    def update_label_window(self, y):
        old_label = self.true_label_win[self.pos_window]
        if old_label != -2:
            self.window_counts[old_label] = self.window_counts.get(old_label, 0) - 1
            if self.window_counts[old_label] == 0:
                del self.window_counts[old_label]

        self.true_label_win[self.pos_window] = y
        self.window_counts[y] = self.window_counts.get(y, 0) + 1
        self.pos_window = (self.pos_window + 1) % self.win_label_size

    def cal_sample_weight(self, prob, y, y_pred):
        if self.instance_seen == 1:
            return 1

        sum_labels = sum(self.window_counts.values())
        all_classes_win = list(self.window_counts.keys())
        num_classes = len(all_classes_win)
        if num_classes == 0 or sum_labels == 0:
            return 1
        if y not in all_classes_win:
            return 1
        for cls in all_classes_win:
            if self.window_counts.get(cls, 0) > 0 and sum_labels > 0:
                self.class_imb_ratio[cls] = (self.window_counts[cls] * num_classes) / sum_labels
            else:
                self.class_imb_ratio[cls] = 0

        true_label = int(y)
        pred_probs = [min(1.0, max(0.0, p)) for p in prob]

        max1 = -1
        max2 = -1
        idx1 = -1
        idx2 = -1
        for i, p in enumerate(pred_probs):
            if p > max1:
                max2, idx2 = max1, idx1
                max1, idx1 = p, i
            elif p > max2:
                max2, idx2 = p, i

        y_c1 = idx1
        y_c2 = idx2 if idx2 != -1 else idx1
        margin = max1 - max2

        if y_c1 == true_label:
            tf = 1
            s = 0
        elif y_c2 == true_label:
            tf = -1
            s = 1
        else:
            tf = -1
            s = 0
        diff = (1 - tf * margin) * math.exp(1 - tf - s)

        imb_log = math.log1p(1 / self.class_imb_ratio[true_label])
        imb_factor = imb_log

        imb_logs = [
            math.log1p(1 / r)
            for r in self.class_imb_ratio.values()
            if r > 0
        ]

        max_imb_log = max(imb_logs) if imb_logs else 1.0

        imb_factor_norm = imb_log / max_imb_log

        final_weight = diff * (1 + self.alpha * imb_factor_norm)
        final_weight = max(self.min_w, min(self.max_w, final_weight))

        return final_weight

    def fit(self, x, y, ord_indices, sample_weight):
        assert len(x) == self.D_
        self.update_label_window(y)
        w = float(sample_weight) if sample_weight is not None else 1.0
        if w <= 0:
            print("w is less than 0")
            return

        self.total_ += w
        self.class_counts_[y] += w

        if y not in self.n_:
            D = self.D_
            self.n_[y] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.mean_[y] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.M2_[y] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))
            self.var_[y] = np.ndarray(D, dtype=np.float64, buffer=np.zeros(D))

            self.discrete_counts_[y] = {}
            self.discrete_n_[y] = np.zeros(D, dtype=np.float64)

            for i in ord_indices:
                self.discrete_counts_[y][i] = defaultdict(float)

        ord_set = self._to_ord_set(ord_indices)

        D_ = self.D_
        n_y = self.n_[y]
        mean_y = self.mean_[y]
        M2_y = self.M2_[y]
        var_y = self.var_[y]
        disc_counts_y = self.discrete_counts_[y]
        disc_n_y = self.discrete_n_[y]
        min_var = self.min_var_

        for i in range(D_):
            xi = x[i]
            if xi is None or np.isnan(xi):
                continue

            if i in ord_set:
                disc_counts_y[i][xi] += w
                disc_n_y[i] += w
            else:
                n_old = n_y[i]
                n_new = n_old + w
                n_y[i] = n_new
                if n_new <= 0:
                    print("n_new is less than 0")
                    continue

                mean_old = mean_y[i]
                delta = xi - mean_old
                mean_new = mean_old + (w / n_new) * delta
                mean_y[i] = mean_new

                M2_y[i] += w * delta * (xi - mean_new)

                if n_new <= 1e-12:
                    var_y[i] = 0.0
                else:
                    var_y[i] = M2_y[i] / n_new

                if var_y[i] < min_var and var_y[i] > 0:
                    min_var = var_y[i]

        self.min_var_ = min_var

    def _update_discrete_feature(self, value, y, feature_idx, weight):
        self.discrete_counts_[y][feature_idx][value] += weight
        self.discrete_n_[y][feature_idx] += weight

    def _update_continuous_feature(self, feature_value, y, feature_idx, weight):
        n_old = self.n_[y][feature_idx]
        n_new = n_old + weight
        self.n_[y][feature_idx] = n_new
        if n_new <= 0:
            print("n_new is less than 0")
            return

        mean_old = self.mean_[y][feature_idx]
        delta = feature_value - mean_old
        mean_new = mean_old + (weight / n_new) * delta
        self.mean_[y][feature_idx] = mean_new

        self.M2_[y][feature_idx] += weight * delta * (feature_value - mean_new)

        if n_new <= 1e-12:
            self.var_[y][feature_idx] = 0.0
        else:
            self.var_[y][feature_idx] = self.M2_[y][feature_idx] / n_new

        if self.var_[y][feature_idx] < self.min_var_ and self.var_[y][feature_idx] > 0:
            self.min_var_ = self.var_[y][feature_idx]

    def handleInstace(self, x, y, ord_indices):
        y_pred, prob = self.predict(x, ord_indices)
        predict_prob = np.array(prob)
        if self.sample_weight:
            sample_weight = self.cal_sample_weight(predict_prob, y, y_pred)
        else:
            sample_weight = 1.0
        self.fit(x, y, ord_indices, sample_weight)

        return y_pred