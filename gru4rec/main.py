import os
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd
import numpy as np
import argparse
import tensorflow.contrib.slim as slim

import model
import evaluation

tf.enable_eager_execution()
tf.executing_eagerly()

PATH_TO_TRAIN = '../../dataset/rsc_2019/train_processed_500k_processed.csv'  
# PATH_TO_TRAIN = '../../dataset/rsc_2019/df_expand.csv' 
PATH_TO_TEST = '../../dataset/rsc_2019/val_processed_1500k.csv'


class Args:
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 3
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
    # hidden_act = 'tanh'  # relu
    hidden_act = 'relu'  # relu
    n_items = -1


def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=100, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=1, type=int)
    parser.add_argument('--test', default=2, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='cross-entropy', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)

    return parser.parse_args()


## records
# 50k train augmented data: 0.625
#

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


if __name__ == '__main__':
    command_line = parseArgs()
    data = pd.read_csv(PATH_TO_TRAIN, sep=',', dtype={'reference': np.int64}, nrows=80000)  # nrows=50000)
    valid = pd.read_csv(PATH_TO_TEST, sep=',', dtype={'reference': np.int64}, nrows=80000)  # nrows=40000)

    ############################## preprocess data ##############################
    # data['length'] = data['step'] / data.groupby('session_id')['step'].transform(len)
    # data = data.loc[(data['length'] >= .5)]
    # print('shape after processing = ', data.shape)
    ############################## preprocess data ##############################

    valid = valid[np.in1d(valid['reference'], data['reference'])]  # Must ensure val in train
    print('train val shape is: ', data.shape, valid.shape)

    args = Args()
    args.n_items = len(data['reference'].unique())
    # args.layers = command_line.layer
    # args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    # args.learning_rate = command_line.lr
    args.is_training = command_line.train
    # args.test_model = command_line.test
    # args.hidden_act = command_line.hidden_act
    # args.final_act = command_line.final_act
    # args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    # print(args.dropout_p_hidden)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    with tf.Session(config=gpu_config) as sess:
        gru = model.GRU4Rec(sess, args)
        model_summary()

        if args.is_training:
            gru.fit(data)

            print('write graph to file...')
            writer = tf.summary.FileWriter('./checkpoint', sess.graph)
            # sess.run(embedding_var.initializer)
            # config = projector.ProjectorConfig()
            # embedding = config.embeddings.add()
            # embedding.tensor_name = embedding_var.name
        else:
            res = evaluation.evaluate_sessions_batch(gru, data, valid)
            print('Recall@25: {}\tMRR@25: {}'.format(res[0], res[1]))
