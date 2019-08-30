import numpy as np
import pandas as pd
# from collections import OrderedDict
# import theano
# from theano import tensor as T
# from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from tqdm import tqdm


def evaluate_sessions_batch(pr, test_data, items=None, cut_off=25, batch_size=100, mode='conservative',
                            session_key='session_id', item_key='reference',
                            time_key='timestamp'):  # mode='standard' 'conservative' 'median' 'tiebreaking'
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    mode : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties, which can mess up the evaluation. 
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates; (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
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
    print('Measuring Recall@{} and MRR@{}'.format(cut_off, cut_off))
    test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx': pr.itemidmap.values, item_key: pr.itemidmap.index}),
                         on=item_key, how='inner')
    test_data.sort_values([session_key, time_key, item_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    iters = np.arange(batch_size).astype(np.int32)
    # pos = np.zeros(min(batch_size, len(session_idx_arr))).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    in_idx = np.zeros(batch_size, dtype=np.int32)
    sampled_items = (items is not None)
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask] - start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]
        for i in range(minlen - 1):
            out_idx = test_data[item_key].values[start_valid + i + 1]
            if sampled_items:
                uniq_out = np.unique(np.array(out_idx, dtype=np.int32))
                preds = pr.predict_next_batch(iters, in_idx, np.hstack([items, uniq_out[~np.in1d(uniq_out, items)]]),
                                              batch_size)
            else:
                preds = pr.predict_next_batch(iters, in_idx, None, batch_size)  # TODO: Handling sampling?
            preds.fillna(0, inplace=True)

            in_idx[valid_mask] = out_idx
            if mode == 'tiebreaking':
                preds += 1e-10 * np.random.rand(*preds.values.shape)
            if sampled_items:
                others = preds.ix[items].values.T[valid_mask].T
                targets = np.diag(preds.ix[in_idx].values)[valid_mask]
                if mode == 'standard':
                    ranks = (others > targets).sum(axis=0) + 1
                elif mode == 'conservative':
                    ranks = (others >= targets).sum(axis=0)
                elif mode == 'median':
                    ranks = (others > targets).sum(axis=0) + 0.5 * ((others == targets).sum(axis=0) - 1) + 1
                elif mode == 'tiebreaking':
                    ranks = (others > targets).sum(axis=0) + 1
                else:
                    raise NotImplementedError
            else:
                if mode == 'standard':
                    ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(
                        axis=0) + 1
                elif mode == 'conservative':
                    ranks = (preds.values.T[valid_mask].T >= np.diag(preds.ix[in_idx].values)[valid_mask]).sum(axis=0)
                elif mode == 'median':
                    ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(
                        axis=0) + 0.5 * ((preds.values.T[valid_mask].T == np.diag(preds.ix[in_idx].values)[
                        valid_mask]).sum(axis=0) - 1) + 1
                elif mode == 'tiebreaking':
                    ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_idx].values)[valid_mask]).sum(
                        axis=0) + 1
                else:
                    raise NotImplementedError
            rank_ok = ranks <= cut_off
            rank_ok_10 = ranks <= 10
            recall += rank_ok_10.sum()
            mrr += ((1.0 / ranks) * rank_ok).sum()
            evalutation_point_count += len(ranks)
            # pos += 1
        start = start + minlen - 1
        mask = np.arange(len(iters))[valid_mask & (end - start <= 1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions) - 1:
                iters[idx] = -1
            else:
                # pos[idx] = 0
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]
    return recall / evalutation_point_count, mrr / evalutation_point_count


def evaluate_gpu(gru, test_data, items=None, session_key='session_id', item_key='reference', time_key='timestamp',
                 cut_off=20,
                 batch_size=100, mode='conservative'):
    if gru.error_during_train: raise Exception
    print('Measuring Recall@{} and MRR@{}'.format(cut_off, cut_off))
    srng = RandomStreams()
    X = T.ivector()
    Y = T.ivector()
    M = T.iscalar()
    C = []
    yhat, H, updatesH = gru.symbolic_predict(X, Y, M, items, batch_size)
    if mode == 'tiebreaking': yhat += srng.uniform(size=yhat.shape) * 1e-10
    if items is None:
        targets = T.diag(yhat.T[Y])
        others = yhat.T
    else:
        targets = T.diag(yhat.T[:M])
        others = yhat.T[M:]
    if mode == 'standard':
        ranks = (others > targets).sum(axis=0) + 1
    elif mode == 'conservative':
        ranks = (others >= targets).sum(axis=0)
    elif mode == 'median':
        ranks = (others > targets).sum(axis=0) + 0.5 * ((others == targets).sum(axis=0) - 1) + 1
    elif mode == 'tiebreaking':
        ranks = (others > targets).sum(axis=0) + 1
    else:
        raise NotImplementedError
    REC = (ranks <= cut_off).sum()
    MRR = ((ranks <= cut_off) / ranks).sum()
    evaluate = theano.function(inputs=[X, Y, M] + C, outputs=[REC, MRR], updates=updatesH, allow_input_downcast=True,
                               on_unused_input='ignore')
    test_data = pd.merge(test_data, pd.DataFrame({'ItemIdx': gru.itemidmap.values, item_key: gru.itemidmap.index}),
                         on=item_key, how='inner')
    test_data.sort_values([session_key, time_key, item_key], inplace=True)
    test_data_items = test_data.ItemIdx.values
    if items is not None:
        item_idxs = gru.itemidmap[items]
    recall, mrr, n = 0, 0, 0
    iters = np.arange(batch_size)
    maxiter = iters.max()
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    finished = False
    cidxs = []
    while not finished:
        minlen = (end - start).min()
        out_idx = test_data_items[start]
        for i in range(minlen - 1):
            in_idx = out_idx
            out_idx = test_data_items[start + i + 1]
            if items is not None:
                y = np.hstack([out_idx, item_idxs])
            else:
                y = out_idx
            rec, m = evaluate(in_idx, y, len(iters), *cidxs)
            recall += rec
            mrr += m
            n += len(iters)
        start = start + minlen - 1
        finished_mask = (end - start <= 1)
        n_finished = finished_mask.sum()
        iters[finished_mask] = maxiter + np.arange(1, n_finished + 1)
        maxiter += n_finished
        valid_mask = (iters < len(offset_sessions) - 1)
        n_valid = valid_mask.sum()
        if n_valid == 0:
            finished = True
            break
        mask = finished_mask & valid_mask
        sessions = iters[mask]
        start[mask] = offset_sessions[sessions]
        end[mask] = offset_sessions[sessions + 1]
        iters = iters[valid_mask]
        start = start[valid_mask]
        end = end[valid_mask]
        if valid_mask.any():
            for i in range(len(H)):
                tmp = H[i].get_value(borrow=True)
                tmp[mask] = 0
                tmp = tmp[valid_mask]
                H[i].set_value(tmp, borrow=True)
    return recall / n, mrr / n


# evaluation with mask impression
def evaluate_sessions(pr, test_data, train_data, items=None, cut_off=25, session_key='session_id',
                      item_key='reference', time_key='timestamp'):
    test_data.sort_values([session_key, time_key], inplace=True)  # sort test set
    items_to_predict = train_data[item_key].unique()  # predict train unique items
    # print('type: ', type(items_to_predict[0]))
    evaluation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0., 0.
    df_out = test_data.copy().reset_index(drop=True)
    df_out = df_out[['user_id', 'session_id', 'timestamp', 'step', 'action_type', 'reference']]
    df_out['prediction'] = pd.Series(np.zeros(shape=(len(df_out))), index=df_out.index)

    # for every element in test set
    for i in tqdm(range(len(test_data))):

        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]

        if prev_sid != sid:  # if sid not eq to -1
            prev_sid = sid  # set prev sid to current sid, then skip, since it is the first one
            # prev_iid = iid
        else:
            if test_data['action_type'].iloc[i] == 'clickout item':
                # predicted score
                preds = pr.predict_next(sid, prev_iid, items_to_predict)
                preds[np.isnan(preds)] = 0
                preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties

                # mask with impression list
                imp_list = test_data['impressions'].iloc[i].split('|')
                mask_imp = np.in1d(items_to_predict, imp_list)
                preds_imp = preds[mask_imp]  # preds_imp is series
                # print('preds_imp = ', preds_imp)
                # print(type(preds_imp[0]))

                # get rank
                try:
                    rank = (preds_imp > preds_imp[iid]).sum() + 1
                except:
                    print('index not in list: ', preds_imp, iid)
                    rank = np.inf

                assert rank > 0  # check the prediction for item_i's position = ran
                # print('rank = ', rank)
                if rank < cut_off:  # & (str(iid) in imp_list)
                    # recall += 1
                    mrr += 1.0 / rank

                if rank < 5:  # set recall@5
                    recall += 1

                evaluation_point_count += 1  # inside clickout item

                item_rec = list(
                    preds_imp.sort_values(ascending=False).index)  # item index to recommend acc to score values

                df_out.loc[i, ['prediction']] = str(item_rec)
                # print('length of item rec = ', len(item_rec))

        prev_iid = iid

        if evaluation_point_count % 100 == 0 and evaluation_point_count != 0:
            # print('evaluate session done = ', i / len(test_data))
            print(evaluation_point_count,
                  ': recall=', recall / evaluation_point_count,
                  ', MRR=', mrr / evaluation_point_count)
        #     df_out.to_csv('./df_out.csv', index=False)

    assert evaluation_point_count > 0
    print('evaluation point = ', evaluation_point_count)
    df_out.to_csv('./baseline_df_out.csv', index=False)
    return recall / evaluation_point_count, mrr / evaluation_point_count


def evaluate_sessions_no_mask_imp(pr, test_data, train_data, items=None, cut_off=25, session_key='session_id',
                                  item_key='reference', time_key='timestamp'):
    test_data.sort_values([session_key, time_key], inplace=True)  # sort test set
    items_to_predict = train_data[item_key].unique()  # predict train unique items
    # print('type: ', type(items_to_predict[0]))
    evalutation_point_count = 0
    prev_iid, prev_sid = -1, -1
    mrr, recall = 0.0, 0.0

    # for every element in test set
    for i in range(len(test_data)):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]

        if prev_sid != sid:  # if sid not eq to -1
            prev_sid = sid  # set prev sid to current sid, then skip, since it is the first one
            prev_iid = iid
        else:
            if test_data['action_type'].iloc[i] == 'clickout item':
                # predicted score
                preds = pr.predict_next(sid, prev_iid, items_to_predict)
                preds[np.isnan(preds)] = 0
                preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties

                # mask with impression list
                # imp_list = test_data['impressions'].iloc[i].split('|')
                # mask_imp = np.in1d(items_to_predict, imp_list)  # length = 1st item, where 2nd item true
                # print(mask_imp, sum(mask_imp))
                # preds_imp = preds[mask_imp]

                # get rank
                rank = (preds > preds[iid]).sum() + 1
                assert rank > 0  # check the prediction for item_i's position = ran
                print('rank = ', rank)
                if rank < cut_off:  # & (str(iid) in imp_list)
                    recall += 1
                    mrr += 1.0 / rank

                evalutation_point_count += 1  # inside clickout item
        prev_iid = iid

    assert evalutation_point_count > 0
    print('evaluation point = ', evalutation_point_count)
    return recall / evalutation_point_count, mrr / evalutation_point_count
