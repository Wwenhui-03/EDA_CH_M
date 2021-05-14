#coding=utf8
from models import *
import dataUtils
import numpy as np
import time
import os
from focal_loss import *
from compile import *


train_path='./data/train'
test_path='./data/test'
dev_percent = 0.05
# Load data
print("Loading data...")
x_, y_, vocabulary, vocabulary_inv,train_size, test_size,sent_length,num_class = dataUtils.load_data(train_path,test_path)
# x_:长度为5952的np.array。（包含5452个训练集和500个测试集）其中每个句子都是padding成长度为37的list（padding的索引为0）
# y_:长度为5952的np.array。每一个都是长度为6的onehot编码表示其类别属性
# vocabulary：长度为8789的字典，说明语料库中一共包含8789各单词。key是单词，value是索引
# vocabulary_inv：长度为8789的list，是按照单词出现次数进行排列。依次为：<PAD?>,\\?,the,what,is,of,in,a....
# test_size:500,测试集大小

# Randomly shuffle data
x, x_test = x_[:-test_size], x_[-test_size:]
y, y_test = y_[:-test_size], y_[-test_size:]
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

x_train, x_dev = x_shuffled[:int(-dev_percent*train_size)], x_shuffled[int(-dev_percent*train_size):]
y_train, y_dev = y_shuffled[:int(-dev_percent*train_size)], y_shuffled[int(-dev_percent*train_size):]

print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))

yf = train(x_train, x_dev, y_train, y_dev, x_test, y_test)
yf.train(sent_length,num_class)
