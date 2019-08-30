import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def split_train_val_2019(train):
    tmax = train['timestamp'].max()
    tmin = train['timestamp'].min()
    session_max_times = train.groupby('session_id')['timestamp'].max()

    # set split date
    session_train = session_max_times[session_max_times < tmax - 86400].index  # train
    session_val = session_max_times[session_max_times >= tmax - 86400].index  # val

    mask_tr = np.in1d(train['session_id'], session_train)  # mask for train set
    mask_val = np.in1d(train['session_id'], session_val)  # rest will be val set
    train_tr = train[mask_tr]  # train set
    val = train[mask_val]  # val set
    val = val[np.in1d(val['reference'], train_tr['reference'])]  # MUST ensure val itemId inside train!!!

    tslength = val.groupby('session_id').size()
    val = val[np.in1d(val['session_id'], tslength[tslength >= 2].index)]  # mask action >=2

    return train_tr, val
    # 0


df = pd.read_csv('D:/pycharm_workspace/dataset/rsc_2019/train.csv', nrows=100000)
idx = df[df['reference'].map(representsInt)].index
df_items = df.loc[idx]  # df items

# split here
train, val = split_train_val_2019(df_items)

train.to_csv('D:/pycharm_workspace/dataset/rsc_2019/train_processed_100k.csv', index=False)
val.to_csv('D:/pycharm_workspace/dataset/rsc_2019/val_processed_100k.csv', index=False)
