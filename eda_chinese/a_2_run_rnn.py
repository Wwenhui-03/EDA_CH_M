# coding: utf-8

from __future__ import print_function

import os
from a_config import *
from methods import *
from numpy.random import seed
from tensorflow import keras
import tensorflow as tf
#gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
#assert len(gpu) == 1
#tf.config.experimental.set_memory_growth(gpu[0], True)

seed(5)
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存8
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config=config)
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from a_config import *
from rnn_model import TRNNConfig, TextRNN
from loader import read_vocab, read_category, batch_iter, process_file, build_vocab
# 4
#base_dir='experiment/weibo'
# 2
#base_dir = 'experiment/hotel'
#base_dir='experiment/weibo_senti'
#base_dir='experiment/waimai'
#train_dir = os.path.join(base_dir, 'data/train_tiny.txt')
#test_dir = os.path.join(base_dir, 'data/test.txt')
#val_dir = os.path.join(base_dir, 'data/val.txt')
#vocab_dir = os.path.join(base_dir, 'data/vocab.txt')

#save_dir = base_dir +'/checkpoints/textrnn/tiny'
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


def train(save_dir,val_dir,train_dir, word_to_id, cat_to_id, config, model, save_path):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = save_dir+'tensorboard/textrnn'
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
                loss_val, acc_val = evaluate(session, x_val, y_val, model)  # todo

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
    print(metrics.classification_report(y_test_cls, y_pred_cls,  target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return acc_test

def run_rnn(train_dir_save, base_dir ,train_path, test_path, categories,num_class):




    #数据来源及存储
    train_dir = train_path
    test_dir = os.path.join(test_path, 'test.txt')
    val_dir = os.path.join(test_path, 'val.txt')
    vocab_dir = os.path.join(base_dir, 'vocab.txt')

    save_dir = train_dir_save + '/checkpoints/textrnn/'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径



    #模型
    print('Configuring RNN model...')
    tf.reset_default_graph()
    config = TRNNConfig(num_class)

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category(categories)
    #categries为['0','1']或['0','1','2','3']
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextRNN(config)

    train(save_dir, val_dir, train_dir, word_to_id, cat_to_id, config, model, save_path)
    acc_test = test(save_path, test_dir, model, word_to_id, cat_to_id, config, categories)
    return acc_test

if __name__ == "__main__":
    for see in range(1):
        seed(see)
        print('seed:', see)
        for a_method in a_methods:
            write_path='outputs_f1_rnn/'
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            filename='outputs_f1_rnn/' + a_method + '_'+str(see) + '_' + get_now_str() + '.txt'
            writer = open(filename, 'a+')
            #for each size dataset
            for size_folder in size_folders:
                writer.write(size_folder + '\n')
                #get all six datasets
                dataset_folders = [size_folder + '/' + s for s in datasets]
                #for storing the performances
                performances = {alpha:[] for alpha in alphas}

                #for each dataset
                for i in range(len(dataset_folders)):
                    #initialize all the variables
                    num_class = num_classes_list[i]
                    dataset_folder = dataset_folders[i]
                    dataset = datasets[i]
                    num_classes = num_classes_list[i]

                    input_size = input_size_list[i]
                    category = categories[i]


                    #test each alpha value

                    for alpha in alphas:

                        base_dir = dataset_folder
                        train_dir_save =dataset_folder + '/train_' + a_method + '_' + str(alpha)
                        train_path = dataset_folder + '/train_' + a_method + '_' + str(alpha) + '.txt'
                        test_path = 'size_data_f1/test/' + dataset
                        acc = run_rnn(train_dir_save, base_dir, train_path, test_path, category, num_class)
                        performances[alpha].append(acc)
                        print(performances)
                writer.write(str(performances) + '\n')
                writer.close()
                writer = open(filename, 'a+')
                for alpha in performances:
                    line = str(alpha) + ' : ' + str(sum(performances[alpha])/len(performances[alpha]))
                    writer.write(line + '\n')
                    print(line)
                print(performances)
        writer.close()
