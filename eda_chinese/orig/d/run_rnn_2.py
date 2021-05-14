# coding: utf-8

from __future__ import print_function

import os
from a_config import *
from methods import *
from numpy.random import seed
from tensorflow import keras
import tensorflow as tf

# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)

seed(5)
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存8
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.compat.v1.Session(config=config)
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics

from b_config import *
from rnn_model import TRNNConfig, TextRNN
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

from sklearn.model_selection import train_test_split
# 4
# base_dir='experiment/weibo'
# 2
# base_dir = 'experiment/hotel'
# base_dir='experiment/weibo_senti'
# base_dir='experiment/waimai'
# train_dir = os.path.join(base_dir, 'data/train_tiny.txt')
# test_dir = os.path.join(base_dir, 'data/test.txt')
# val_dir = os.path.join(base_dir, 'data/val.txt')
# vocab_dir = os.path.join(base_dir, 'data/vocab.txt')

# save_dir = base_dir +'/checkpoints/textrnn/tiny'
# save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


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


def train(save_dir, val_dir, train_dir, word_to_id, cat_to_id, config, model, save_path, percent_dataset,see):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = save_dir + 'tensorboard/textrnn'
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
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    return acc_test


def run_rnn(train_dir_save, base_dir, train_path, test_path, categories, num_class, increment,see, model_output_path):
    # 数据来源及存储
    train_dir = train_path
    test_dir = test_path
    val_dir = os.path.join(base_dir, 'val.txt')
    vocab_dir = os.path.join(base_dir, 'vocab.txt')

    save_dir = train_dir_save + '/checkpoints/textrnn/'
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

    # 模型
    print('Configuring RNN model...')
    tf.reset_default_graph()
    config = TRNNConfig(num_class)

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category(categories)
    # categries为['0','1']或['0','1','2','3']
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    model = TextRNN(config)

    train(save_dir, val_dir, train_dir, word_to_id, cat_to_id, config, model, save_path, increment,see)
    acc_test = test(save_path, test_dir, model, word_to_id, cat_to_id, config, categories)
    return acc_test,model


from methods import *
from numpy.random import seed
from keras import backend as K
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re

seed(0)


################################
#### get dense layer output ####
################################

# getting the x and y inputs in numpy array form from the text file
def train_x(train_txt, word2vec_len, input_size, word2vec):
    # read in lines
    train_lines = open(train_txt, 'r').readlines()
    num_lines = len(train_lines)

    x_matrix = np.zeros((num_lines, input_size, word2vec_len))

    # insert values
    for i, line in enumerate(train_lines):

        parts = line[:-1].split('\t')
        parts[0] = re.sub(r'\ufeff', "", parts[0])
        label = int(parts[0])
        sentence = parts[1]

        # insert x
        words = sentence.split(' ')
        words = words[:x_matrix.shape[1]]  # cut off if too long
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i, j, :] = word2vec[word]

    return x_matrix


def get_dense_output(model_checkpoint, file, num_classes,model):
    x = train_x(file, word2vec_len, input_size, word2vec)



    get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[4].output])
    layer_output = get_3rd_layer_output([x])[0]

    return layer_output


def get_tsne_labels(file):
    labels = []
    alphas = []
    lines = open(file, 'r').readlines()
    for i, line in enumerate(lines):
        parts = line[:-1].split('\t')
        _class = int(parts[0])
        alpha = i % 10
        labels.append(_class)
        alphas.append(alpha)
    return labels, alphas


def get_plot_vectors(layer_output):
    tsne = TSNE(n_components=2).fit_transform(layer_output)
    return tsne


def plot_tsne(tsne, labels, output_path):
    label_to_legend_label = {'outputs_f4/hotel_tsne.png': {0: 'Con (augmented)',
                                                           100: 'Con (original)',
                                                           1: 'Pro (augmented)',
                                                           101: 'Pro (original)'},
                             'outputs_f4/tnews_tsne.png': {0: 'Description (augmented)',
                                                           100: 'Description (original)',
                                                           1: 'Entity (augmented)',
                                                           101: 'Entity (original)',
                                                           2: 'Abbreviation (augmented)',
                                                           102: 'Abbreviation (original)',
                                                           3: 'Human (augmented)',
                                                           103: 'Human (original)',
                                                           4: 'Location (augmented)',
                                                           104: 'Location (original)',
                                                           5: 'Number (augmented)',
                                                           105: 'Number (original)'}}

    plot_to_legend_size = {'outputs_f4/hotel_tsne.png': 11, 'outputs_f4/tnews_tsne.png': 6}

    labels = labels.tolist()
    big_groups = [label for label in labels if label < 100]
    big_groups = list(sorted(set(big_groups)))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500']
    fig, ax = plt.subplots()

    for big_group in big_groups:

        for group in [big_group, big_group + 100]:

            x, y = [], []

            for j, label in enumerate(labels):
                if label == group:
                    x.append(tsne[j][0])
                    y.append(tsne[j][1])

            # params
            color = colors[int(group % 100)]
            marker = 'x' if group < 100 else 'o'
            size = 1 if group < 100 else 27
            legend_label = label_to_legend_label[output_path][group]

            ax.scatter(x, y, color=color, marker=marker, s=size, label=legend_label)
            plt.axis('off')

    legend_size = plot_to_legend_size[output_path]
    plt.legend(prop={'size': legend_size})
    plt.savefig(output_path, dpi=1000)
    plt.clf()


if __name__ == "__main__":

    # parameters
    dataset_folders = ['increment_datasets_f2/hotel', 'increment_datasets_f2/tnews']
    output_paths = ['outputs_f4/hotel_aug_cnn.h5', 'outputs_f4/tnews_aug_cnn.h5']
    # number of output classes
    num_classes_list = [2, 15]
    categories = [['0', '1'],
                  ['100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112', '113', '114', '115',
                   '116']]
    input_size_list = [25, 25]
    datasets = ['hotel', 'tnews']
    # word2vec dictionary
    word2vec_len = 300
    for see in range(5):
        for i, dataset_folder in enumerate(dataset_folders):
            writer = open('output_f4/' + str(see) + '.csv', 'a+')
            num_classes = num_classes_list[i]
            input_size = input_size_list[i]
            output_path = output_paths[i]
            train_orig = dataset_folder + '/train_aug_st.txt'
            test_path = dataset_folder + '/test.txt'
            word2vec_pickle = dataset_folder + '/word2vec.p'
            word2vec = load_pickle(word2vec_pickle)
            category = categories[i]
            num_classes = num_classes_list[i]
            # train model and save
            train_dir_save = dataset_folder + '/train_' + '_' + str(dataset_folder)
            base_dir = dataset_folder
            acc, model = run_rnn(train_dir_save, base_dir, train_orig, test_path, category, num_classes, 1, output_path)
            writer.write(dataset_folder + ',' + str(acc) + '\n')

            print(dataset_folder, acc)
            dataset=datasets[i]
            # load parameters
            model_checkpoint = 'outputs_f4/' + dataset + '.h5'
            file = 'special_f4/' + dataset + '/test_short_aug.txt'
            num_classes = num_classes_list[i]
            word2vec_pickle = 'special_f4/' + dataset + '/word2vec.p'
            word2vec = load_pickle(word2vec_pickle)

            # do tsne
            layer_output = get_dense_output(model_checkpoint, file, num_classes, model)

            print(layer_output.shape)
            writer.write(str(layer_output.shape) + ','  + '\n')
            t = get_plot_vectors(layer_output)

            labels, alphas = get_tsne_labels(file)

            print(labels, alphas)
            writer.write(str(labels) +','+str(alphas)+ ',' + '\n')
            writer.close()
            writer = open("output_f4/new_tsne"+str(see)+".txt", 'w')

            label_to_mark = {0: 'x', 1: 'o',100 :'100',101 :'101',102 :'102',103 :'103',104 :'104',105 :'105',106 :'106',107 :'107',108 :'108',109 :'109',110 :'110',111 :'111',112 :'112',113 :'113',114 :'114',115 :'115',116 :'116'}

            for j, label in enumerate(labels):
                alpha = alphas[j]
                line = str(t[j, 0]) + ' ' + str(t[j, 1]) + ' ' + str(label_to_mark[label]) + ' ' + str(alpha / 10)
                writer.write(line + '\n')




