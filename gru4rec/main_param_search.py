import os
import tensorflow as tf

tf.enable_eager_execution()

import pandas as pd
import numpy as np
import argparse
import random

import model
import evaluation

PATH_TO_TRAIN = '../../dataset/rsc_2019/train_processed_500k_processed.csv'  # 
# PATH_TO_TRAIN = '../../dataset/rsc_2019/df_expand.csv'  # 
PATH_TO_TEST = '../../dataset/rsc_2019/val_processed_1500k.csv'


class Args:
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 1
    batch_size = 50
    dropout_p_hidden = 1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'session_id'
    item_key = 'reference'
    time_key = 'timestamp'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'relu'
    n_items = -1


def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='relu', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)

    return parser.parse_args()


def find_param():
    param_grid = {
        'layers': [1, ],
        # 'rnn_size': list(range(50, 150, 5)),
        'rnn_size': [100],
        'learning_rate': list(np.logspace(np.log10(0.001), np.log10(0.01), base=10, num=1000)),
    }
    # Randomly sample from dict
    params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
    return params


if __name__ == '__main__':
    tf.reset_default_graph()
    data = pd.read_csv('./train.csv', sep=',', dtype={'reference': np.int64}, nrows=50000)
    valid = pd.read_csv('./val.csv', sep=',', dtype={'reference': np.int64}, nrows=50000)
    print('shape before in1d: ', data.shape, valid.shape)
    valid = valid[np.in1d(valid['reference'], data['reference'])]  # Must ensure val in train
    print('shape after: ', data.shape, valid.shape)

    args = Args()
    args.n_items = len(data['reference'].unique())
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else 0.5

    # if not os.path.exists(args.checkpoint_dir):
    #     os.mkdir(args.checkpoint_dir)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

mrr_list = []

for i in range(10):
    params = find_param()
    print('param searching round =', i)
    print(params)
    args.layers = params['layers']
    args.rnn_size = params['rnn_size']
    args.learning_rate = float(params['learning_rate'])  # cast to float

    tf.reset_default_graph()
    with tf.Session(config=gpu_config) as sess:
        args.is_training = True
        gru = model.GRU4Rec(sess, args)
        gru.fit(data)

    tf.reset_default_graph()  # reset graph 
    with tf.Session(config=gpu_config) as sess:
        args.is_training = False
        gru = model.GRU4Rec(sess, args)  # instantiate again
        res = evaluation.evaluate_sessions_batch(gru, data, valid)
        print('Recall@25: {}\tMRR@25: {} \n'.format(res[0], res[1]))
