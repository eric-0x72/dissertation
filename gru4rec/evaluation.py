import numpy as np
import pandas as pd


def evaluate_sessions_batch(model, train_data, test_data, cut_off=25, batch_size=50, session_key='session_id',
                            item_key='reference', time_key='timestamp'):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    Parameters
    --------
    model : A trained GRU4Rec model.
    train_data : It contains the transactions of the train set. In evaluation phrase, this is used to build item-to-id map.
    test_data : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)

    '''
    model.predict = False

    # Build itemidmap from train data.
    itemids = train_data[item_key].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)
    # print(len(itemidmap))  # 2925 items

    test_data.sort_values([session_key, time_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    # offset session: unique session's action size cumsum
    # offset sessions: [0   2   8  31  62  63  64  80 116 120 122 156 158 162 163 166 177 178
    #                  180 191 194 198 199 263 273 276 309 310 322 339 340 565 567 569 574 576
    #                  644 658 659 672 673 687 688 728 730 735 736 738 740 742 841 843 845]

    evaluation_point_count = 0
    mrr, recall = 0.0, 0.0

    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1

    iters = np.arange(batch_size).astype(np.int32)  # initialise iters [0 to 49]
    maxiter = iters.max()  # max is 49
    # print('iters = ', iters)
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]  # count 50 session in one batch, from offset sessions
    # print('start is :', start); print('end is :', end)

    in_idx = np.zeros(batch_size, dtype=np.int32)  # 50 zeros
    np.random.seed(42)

    df_out = test_data.copy().reset_index(drop=True)
    df_out = df_out[['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference', 'impressions']]
    df_out['prediction'] = pd.Series(np.zeros(shape=(len(df_out))), index=df_out.index)  # create prediction column

    arr = np.array([0] * len(test_data))

    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break

        start_valid = start[valid_mask]
        minlen = (end[valid_mask] - start_valid).min()  # min session length
        in_idx[valid_mask] = test_data[item_key].values[start_valid]  # item index at beginning of session

        # for min no. of actions in current batch
        for i in range(minlen - 1):
            imp = pd.Series(test_data['impressions'].values[start_valid + i + 1])
            out_idx = test_data[item_key].values[start_valid + i + 1]
            mask_batch_clickout = test_data['action_type'].values[start_valid + i + 1] == 'clickout item'  # here!!!
            preds = model.predict_next_batch(iters,
                                             in_idx,
                                             itemidmap,
                                             batch_size)  # preds: shape is 2925*50, 2925 items in 50 sessions
            preds.fillna(0, inplace=True)
            preds_idx = preds.index.values.tolist()

            in_idx[valid_mask] = out_idx  # next batch in equal to out_idx
            # score of pred ranking compare with ground truth
            # print(preds.loc[in_idx].values.shape)

            # print('added', preds.iloc[:, 0])
            ids = start_valid + i + 1
            # test_data[ids] = 0
            # order = [list(preds.iloc[:, j].sort_values(ascending=False).index) for j in range(len(ids))]

            # li = []
            # # imp = pd.Series(test_data['impressions'].values[start_valid + i + 1])
            # for j in range(len(ids)):
            #     # print(j, ids[j])
            #     if test_data['action_type'].iloc[ids[j]] == 'clickout item':
            #
            #         # print('impression = ', test_data['impressions'].iloc[ids[j]])
            #         split_str = (test_data['impressions'].iloc[ids[j]].split('|'))
            #         split_num = list(map(int, split_str))
            #         mask = np.in1d(list(preds.iloc[:, j]), split_num)
            #         col_k = preds.iloc[:, j]
            #         preds.iloc[:, j] = np.multiply(col_k, mask)  # set rest values to 0
            #         # print(type(preds.iloc[:, j].sort_values(ascending=False).index.values))
            #         li.append(preds.iloc[:, j].sort_values(ascending=False).index.values[:25])
            #     else:
            #         li.append(0)

            # df_out.loc[ids, ['prediction']] = li

            # df_out.to_csv('./gru4rec_df_out.csv', index=False)
            # break

            # preds.to_pickle('./preds.pickle', )

            # mask not clickout with 0; k is batch size 50, also imp mask batch
            for k in range(len(mask_batch_clickout)):
                if not mask_batch_clickout[k]:  # action is not clickout, make that col zero
                    preds.iloc[:, k] = [0] * preds.shape[0]  # col of 2925*1 zeros

                else:  # if item is clickout, then mask with impressions
                    if isinstance(imp.iloc[k], str):  # check if string, before splitting
                        split_str = (imp.iloc[k].split('|'))
                        split_num = list(map(int, split_str))
                        mask = np.in1d(preds_idx, split_num)
                        col_k = preds.iloc[:, k]
                        preds.iloc[:, k] = np.multiply(col_k, mask)

                        # ids = start_valid + i + 1
                        # item_rec = list(preds.iloc[:, k].sort_values(ascending=False).index)

            ranks = (preds.values.T[valid_mask].T > np.diag(preds.loc[in_idx].values)[valid_mask]).sum(axis=0) + 1
            ranks = ranks[mask_batch_clickout]  # only take clickout here!!
            rank_ok = ranks < cut_off
            rank_ok_5 = ranks < 5
            recall += rank_ok_5.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evaluation_point_count += len(ranks)

            if evaluation_point_count % 100 == 0 and evaluation_point_count != 0:
                print('eval point count=', evaluation_point_count,
                      ': recall=', recall / evaluation_point_count,
                      ', MRR=', mrr / evaluation_point_count)

        start = start + minlen - 1  # start of next batch
        mask = np.arange(len(iters))[valid_mask & (end - start <= 1)]  # mask for sessions with 1 action left

        for idx in mask:  # fill in sessions
            maxiter += 1

            if maxiter >= len(offset_sessions) - 1:  # no more sessions to add
                iters[idx] = -1

            else:  # add new session into iter
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]

    # df_out.to_pickle('./df_out.pickle')
    #
    # # doing ...
    # for k in range(len(df_out)):
    #     if df_out['action_type'].loc[k] == 'clickout item':
    #         try:
    #             split_str = (df_out['impressions'].loc[k].split('|'))
    #             split_num = list(map(int, split_str))
    #
    #             preds_idx = df_out['prediction'].loc[k]
    #             # print('preds_idx = ', preds_idx)
    #
    #             mask = np.in1d(preds_idx, split_num)

    #             # df_out['prediction'].loc[k] = df_out['prediction'].loc[k][mask]
    #             print(arr[k][mask])
    #             df_out['prediction'].loc[k] = arr[k][mask]
    #
    #             print(df_out['prediction'].loc[k])
    #         except:
    #             df_out['prediction'].loc[k] = 0

    print('evaluation point count = ', evaluation_point_count)
    df_out.to_csv('./gru4rec_df_out.csv', index=False)
    return recall / evaluation_point_count, mrr / evaluation_point_count
