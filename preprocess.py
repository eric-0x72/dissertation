import pandas as pd
import numpy as np
from tqdm import tqdm
import random

df = pd.read_csv('../dataset/rsc_2019/train_processed_500k.csv', )
df = df[['session_id', 'timestamp', 'action_type', 'reference', 'impressions']]


def preprocess(df):
    gp = df.groupby('session_id').groups
    cutoff = 7

    result_ids = pd.Series()
    sess_id_new = pd.Series()

    for k, v in tqdm(gp.items()):  # k is session_id, v is index
        # print(list(v))
        # print('session =', k, ', session len =', len(v))

        for i in range(1, cutoff + 1):
            random.seed(50)
            rand = str(random.randint(1, 999))

            ids = v[-i:]
            mask = np.random.choice([False, True], size=(len(ids),), p=[1./5, 1 - 1./5])
            # df_expand = df_expand.append(df.loc[ser[mask]])
            masked_ids = ids[mask]

            result_ids = result_ids.append(pd.Series(masked_ids))
            sess_id_new = sess_id_new.append(pd.Series([k + rand] * (len(masked_ids))))
            # session_id + rand = new session identifier

    df_result = df.loc[result_ids].reset_index()
    df_result['session_id'] = sess_id_new.reset_index(drop=True)

    return df_result


df_processed = preprocess(df)
print(df_processed)

df.to_csv('../dataset/rsc_2019/train_processed_500k_processed.csv')
