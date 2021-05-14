#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from b_config import *
from methods import *
from numpy.random import seed
from tensorflow import keras
import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
#assert len(gpu) == 1
#tf.config.experimental.set_memory_growth(gpu[0], True)
from sklearn.model_selection import train_test_split
seed(5)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config=config)

import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from cnn_model import TCNNConfig, TextCNN
from loader import read_vocab, read_category, batch_iter, process_file, build_vocab
from sklearn.model_selection import train_test_split
#base_dir = 'experiment/hotel'
#base_dir='experiment/weibo'
#base_dir='experiment/weibo_senti'
#base_dir='experiment/waimai'
#train_dir = os.path.join(base_dir, 'data/train_full.txt')
#test_dir = os.path.join(base_dir, 'data/test.txt')
#val_dir = os.path.join(base_dir, 'data/val.txt')
#vocab_dir = os.path.join(base_dir, 'data/vocab.txt')

#save_dir = base_dir +'/checkpoints/textcnn/full'
#save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob, model):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_, model):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0, model)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(save_dir,val_dir,train_dir, word_to_id, cat_to_id, config, model, save_path,percent_dataset,see):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = save_dir+'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    ####此处需要进行修改，多次实验取平均值###############################################################################################################
    #x_train = x_train[:int(percent_dataset * len(x_train))]
    #y_train = y_train[:int(percent_dataset * len(y_train))]
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, train_size=percent_dataset, random_state=see)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob, model)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val, model)   # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test(save_path, test_dir, model, word_to_id, cat_to_id, config, categories):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test, model)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    return acc_test

def run_rnn(train_dir_save, base_dir ,train_path, test_path, categories,num_class,increment,see):




    #数据来源及存储
    train_dir = train_path
    test_dir = test_path
    val_dir = os.path.join(base_dir, 'val.txt')
    vocab_dir = os.path.join(base_dir, 'vocab.txt')

    save_dir = train_dir_save + '/checkpoints/textcnn/'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径



    print('Configuring CNN model...')
    tf.reset_default_graph()
    config = TCNNConfig(num_class)

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category(categories)
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextCNN(config)

    train(save_dir, val_dir, train_dir, word_to_id, cat_to_id, config, model, save_path, increment,see)
    acc_test = test(save_path, test_dir, model, word_to_id, cat_to_id, config, categories)
    return acc_test


if __name__ == "__main__":
    # get the accuracy at each increment
    orig_accs = {dataset: {} for dataset in datasets}
    aug_accs = {dataset: {} for dataset in datasets}
    filename = 'outputs_f2_cnn/' + get_now_str()
    write_path='outputs_f2_cnn/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    # for each dataset
    for see in range(5):
        writer = open(filename +str(see)+ '.csv', 'a+')
        for i, dataset_folder in enumerate(dataset_folders):

            dataset = datasets[i]
            num_classes = num_classes_list[i]
            input_size = input_size_list[i]
            train_orig = dataset_folder + '/train_orig.txt'
            train_aug_st = dataset_folder + '/train_aug_st.txt'
            test_path = dataset_folder + '/test.txt'
            category = categories[i]

            for increment in increments:
                # calculate augmented accuracy
                base_dir = dataset_folder
                train_dir_save = dataset_folder + '/train_' +str(see)+ '_' + str(increment)
                aug_acc = run_rnn(train_dir_save, base_dir, train_aug_st, test_path, category, num_classes, increment,see)
                aug_accs[dataset][increment] = aug_acc

                # calculate original accuracy
                orig_acc = run_rnn(train_dir_save, base_dir, train_orig, test_path, category, num_classes, increment,see)
                orig_accs[dataset][increment] = orig_acc

                print(dataset, increment, orig_acc, aug_acc)
                writer.write(dataset + ',' + str(increment) + ',' + str(orig_acc) + ',' + str(aug_acc) + '\n')
                writer.close()
                writer = open(filename +str(see)+'.csv', 'a+')
                gc.collect()

        print(orig_accs, aug_accs)
