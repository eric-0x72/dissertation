import numpy as np
import pandas as pd
from tqdm import tqdm

PATH = './gru4rec_BP/gru4rec_df_out.csv'
test_data = pd.read_csv(PATH, sep=',', dtype={'reference': np.int64}, nrows=1000)


# evaluation with mask impression
def evaluate_sessions(test_data, items=None, cut_off=25, session_key='session_id',
                      item_key='reference', time_key='timestamp'):
    test_data.sort_values([session_key, time_key], inplace=True)  # sort test set
    evalutation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0., 0.

    # for every element in test set
    for i in tqdm(range(len(test_data))):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]

        if prev_sid != sid:  # if sid not eq to -1
            prev_sid = sid  # set prev sid to current sid, then skip, since it is the first one
            # prev_iid = iid
        else:
            if test_data['action_type'].iloc[i] == 'clickout item':
                preds_imp = test_data['reference'].iloc[i]

                # get rank
                try:
                    rank = (preds_imp > preds_imp[iid]).sum() + 1
                except:
                    print('index not in list: ', preds_imp, iid)
                    rank = np.inf

                assert rank > 0  # check the prediction for item_i's position = ran
                # print('rank = ', rank)
                if rank < cut_off:  # & (str(iid) in imp_list)
                    recall += 1
                    mrr += 1.0 / rank

                evalutation_point_count += 1  # inside clickout item
                print('preds_imp = ', preds_imp,)
                item_rec = list(
                    preds_imp.sort_values(ascending=False).index)  # item index to recommend acc to score values

                print('item rec = ', item_rec)

        prev_iid = iid

        if i % 1000 == 0 and i != 0:
            # print('evaluate session done = ', i / len(test_data))
            print(recall / evalutation_point_count, mrr / evalutation_point_count)

    assert evalutation_point_count > 0
    print('evaluation point = ', evalutation_point_count)
    return recall / evalutation_point_count, mrr / evalutation_point_count


evaluate_sessions(test_data=test_data)
