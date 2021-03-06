#coding:utf8
from collections import Counter
import itertools
import numpy as np
import re
import chardet
import jieba

# 获得纯文本
def clean_str(line):

    line = line.replace("\t", " ")
    line = line.replace("\n", " ")

    line = re.sub(' +', ' ', line)  # 删除多余空格

    if (len(line) > 0):
        if line[0] == ' ':
            line = line[1:]
    return line.strip()

def load_data_and_labels(train_path,test_path):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    #folder_prefix = 'data/'
    #测试
    #增加了一个'rb'

    x_train = list(open(train_path, 'rb').readlines())
    x_test = list(open(test_path, 'rb').readlines())
    #x_train = list(open(folder_prefix + "train", 'rb').readlines())
    #x_test = list(open(folder_prefix + "test", 'rb').readlines())
    test_size = len(x_test)
    train_size = len(x_train)
    x_text = x_train + x_test
    #修改
    le = len(x_text)
    for i in range(le):
        encode_type = chardet.detect(x_text[i])
        x_text[i] = x_text[i].decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量）
    #修改
    y = [s.split(' ')[0].split('\t')[0] for s in x_text]
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [ " ".join(jieba.cut(sent)) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    # Generate labels
    all_label = dict()
    for label in y:
        if not label in all_label:
            all_label[label] = len(all_label) + 1
    num_class=len(all_label)
    one_hot = np.identity(len(all_label))
    y = [one_hot[ all_label[label]-1 ] for label in y]
    return [x_text, y,train_size, test_size,num_class]

def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return [padded_sentences,sequence_length]

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    # vocabulary_inv=['<PAD/>', 'the', ....]
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    # vocabulary = {'<PAD/>': 0, 'the': 1, ',': 2, 'a': 3, 'and': 4, ..}
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data(train_path,test_path):
    """
    Loads and preprocessed data
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels,train_size, test_size,num_class = load_data_and_labels(train_path,test_path)
    sentences_padded ,sent_length= pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv,train_size, test_size,sent_length,num_class]

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index > data_size:
                end_index = data_size
                start_index = end_index - batch_size
            yield shuffled_data[start_index:end_index]


