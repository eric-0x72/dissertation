import pandas as pd
import numpy as np
import pprint
import itertools
import random

# df = pd.read_csv('../dataset/rsc_2019/train_processed_1500k.csv', nrows=80000)
# df = pd.read_csv('../../dataset/rsc_2019/item_metadata.csv', nrows=10)
# print(df.head())


N_FOLDS = 5
MAX_EVALS = 5


def find_param():
    # Hyperparameter grid
    param_grid = {
        'layers': [1, 2, 3],
        'rnn_size': list(range(50, 150, 5)),
        'n_epochs': [3],
        'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=1000)),
    }
    # a = random.sample(param_grid['layers'], 1)[0]
    # rnn_size = random.

    # Randomly sample from dictionary
    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    return random_params


t = find_param()
# print(t['layers'])
print(t)
