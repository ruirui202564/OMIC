import random
import os
import numpy as np

#--------------Seed--------------#
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_ord_indices(X):
    max_ord=10
    indices = []
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) <= max_ord:
            indices.append(i)
    return indices

def random_perm_n(n):
    np.random.seed(1)
    permutations = []
    for i in range(10):
        perm = np.arange(n)
        if i == 0:
            permutations.append(perm)
        else:
            np.random.shuffle(perm)
            permutations.append(perm)

    return permutations

def random_perm_n_old(n):
    np.random.seed(1)
    permutations = []
    for _ in range(10):
        perm = np.arange(n)
        np.random.shuffle(perm)
        permutations.append(perm)

    return permutations