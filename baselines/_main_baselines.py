import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from evaluation import evaluate_sessions, evaluate_sessions_no_mask_imp
from knn import cknn, scknn, vmknn, smf
# import gru4rec
import baselines
from tqdm import tqdm


def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


# df = pd.read_csv('../dataset/rsc_2019/train.csv', nrows=50000)  # 50k
# # if 60k, get 48329, 694 as train val
# idx_item = df[df['reference'].map(representsInt)].index
# df_1 = df.loc[idx_item]


# NON-TEST set split for 2019
def split_train_val_2019(train):
    tmax = train['timestamp'].max()
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


# split here
# train, val = split_train_val_2019(df_1)
# train.to_csv('../dataset/rsc_2019/train_processed.csv')
# val.to_csv('../dataset/rsc_2019/val_processed.csv')

# train = pd.read_csv('D:/pycharm_workspace/dataset/rsc_2019/train_processed_1500k.csv', )  # nrows=200000)
# val = pd.read_csv('D:/pycharm_workspace/dataset/rsc_2019/val_processed_1500k.csv', )  #f nrows=200000)
train = pd.read_csv('./train.csv')  # 1500k
val = pd.read_csv('val.csv')  # 153k
print('train val shape before = ', train.shape, val.shape)

train['reference'] = train['reference'].apply(str)
val['reference'] = val['reference'].apply(str)  # cast csv reference back to string, otherwise will be int
# val = val[np.in1d(val['reference'], train['reference'])]  # MUST ensure val itemId inside train!!! ## no need if already done
print('train val shape after =', train.shape, val.shape)
train.to_csv('./train.csv')
val.to_csv('./val.csv')

# @1000 evlauation point!!
# pr = baselines.Pop()  ## pop = recall 0.3271428571428571, MRR 0.24147906462249372
# pr = baselines.ItemKNN() ## iknn = recall 0.23833333333333334, MRR 0.21148592950108158
# pr = scknn.SeqContextKNN(k=200) ## seq-sknn, 0.7011111111111111 0.6245173847048702
# pr = cknn.ContextKNN(k=200)  ## sknn = 0.6936363636363636 0.6155647995716955
# pr = smf.SessionMF()  ## smf = recall 0.3142857142857143, MRR= 0.2455139557537925
# pr = baselines.SessionPop()  ## session pop = 0.6972727272727273 0.6368818669105739
pr = baselines.BPR()  # recall 0.6302272727272727, MRR 0.5462958818263384

pr.fit(train)
print('val shape = ', val.shape)
print(evaluate_sessions(pr, val, train))
