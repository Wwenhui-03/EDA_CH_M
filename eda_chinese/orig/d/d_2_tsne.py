from methods import *
from numpy.random import seed
from keras import backend as K
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
from tensorflow.keras.models import load_model
from pandas import Series
import tensorflow as tf

seed(0)

import tensorflow as tf
import keras


global graph, sess


graph = tf.get_default_graph()
sess = keras.backend.get_session()

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
        seg_list = jieba.cut(sentence)
        seg_list = " ".join(seg_list)
        words = seg_list.split(' ')
        words = words[:x_matrix.shape[1]]  # cut off if too long
        for j, word in enumerate(words):
            if word in word2vec:
                x_matrix[i, j, :] = word2vec[word]

    return x_matrix


def get_dense_output(model_checkpoint, file, num_classes):
    x = train_x(file, word2vec_len, input_size, word2vec)

    model = load_model(model_checkpoint)

    get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[4].output])
    layer_output = get_3rd_layer_output([x])[0]

    return layer_output


def get_tsne_labels(file):
    labels = []
    alphas = []
    lines = open(file, 'r').readlines()
    for i, line in enumerate(lines):
        line = line.replace('\ufeff','')
        parts = line[:-1].split('\t')
        if(len(parts[0])>0):
            _class = int(parts[0])
            alpha = i % 10
            labels.append(_class)
            alphas.append(alpha)
    return labels, alphas


def get_plot_vectors(layer_output):
    tsne = TSNE(n_components=2).fit_transform(layer_output)
    return tsne


def plot_tsne(tsne, labels, output_path):
    label_to_legend_label = {'output_f4/hotel_tsne.png': {0: 'Con (augmented)',
                                                           100: 'Con (original)',
                                                           1: 'Pro (augmented)',
                                                           101: 'Pro (original)'},
                             'output_f4/tnews_tsne.png': { 0: '100 (augmented)',
                                                           100: '100 (original)',
                                                           1: '101(augmented)',
                                                           101: '101 (original)',
                                                           2: '102 (augmented)',
                                                           102: '102 (original)',
                                                           3: '103 (augmented)',
                                                           103: '103 (original)',
                                                           4: '104 (augmented)',
                                                           104: '104 (original)',
                                                           5: '105 (augmented)',
                                                           105: '105 (original)',
                                                           6: '106 (augmented)',
                                                           106: '106 (original)',
                                                           7: '107(augmented)',
                                                           107: '107 (original)',
                                                           8: '108 (augmented)',
                                                           108: '108 (original)',
                                                           9: '109 (augmented)',
                                                           109: '109 (original)',
                                                           10: '110 (augmented)',
                                                           110: '110 (original)',
                                                           11: '111 (augmented)',
                                                           111: '111 (original)',
                                                           12: '112 (augmented)',
                                                           112: '112 (original)',
                                                           13: '113(augmented)',
                                                           113: '113 (original)',
                                                           14: '114 (augmented)',
                                                           114: '114 (original)',
                                                           15: '115 (augmented)',
                                                           115: '115 (original)',
                                                           16: '116 (augmented)',
                                                           116: '116 (original)'
                                                           },
                             'output_f4/fudan_tsne.png': {19: 'Con (augmented)',
                                                          119: 'Con (original)',
                                                          31: 'Pro (augmented)',
                                                          131: 'Pro (original)',
                                                          34: 'Con (augmented)',
                                                          134: 'Con (original)',
                                                          39: 'Pro (augmented)',
                                                          139: 'Pro (original)'}
                             }

    plot_to_legend_size = {'output_f4/hotel_tsne.png': 6, 'output_f4/tnews_tsne.png': 2,'output_f4/fudan_tsne.png': 4}

    labels = labels.tolist()
    big_groups = [label for label in labels if label < 100]
    big_groups = list(sorted(set(big_groups)))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ff1493', '#FF4500','#008B8B','#D2691E','#228B22','#90EE90','#BA55D3','#48D1CC','#C0C0C0','#0000CD']
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
            # 显示图例

            #box = ax.get_position()
            #ax.set_position([box.x0, box.y0, box.width, box.height])
            ax.scatter(x, y, color=color, marker=marker, s=size, label=legend_label)
            plt.axis('off')

    legend_size = plot_to_legend_size[output_path]
    plt.legend(prop={'size': legend_size})
    plt.savefig(output_path, dpi=1000)
    plt.clf()


if __name__ == "__main__":

    # global variables
    word2vec_len = 300


    datasets = ['hotel','tnews']
    input_size = 25
    num_classes_list = [2, 15, 4]
    categories = [['0', '1'],
                  ['100', '101', '102', '103', '104', '106', '107', '108', '109', '110', '112', '113', '114', '115',
                   '116'], ['19', '31', '34', '39']]

    for i, dataset in enumerate(datasets):
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # load parameters
        model_checkpoint = 'output_f4/' + dataset + '.h5'
        file = 'special_f4/' + dataset + '/test_short_aug.txt'
        num_classes = num_classes_list[i]
        word2vec_pickle = 'special_f4/' + dataset + '/word2vec.p'
        word2vec = load_pickle(word2vec_pickle)

        # do tsne
        with sess.as_default():
            with graph.as_default():
                layer_output = get_dense_output(model_checkpoint, file, num_classes)
                print(layer_output.shape)
                t = get_plot_vectors(layer_output)

                labels, alphas = get_tsne_labels(file)
                if(dataset=='tnews'):
                    labels = Series(labels)-100
                else:
                    labels = Series(labels)
                plot_tsne(t, labels, 'output_f4/'+dataset+'_tsne.png')


                #print(labels, alphas)

                #writer = open("output_f4/new_tsne.txt", 'w')

                #label_to_mark = {0: 'x', 1: 'o',2:'*',3:'@',4:'&',5:'#',6:'$',7:'+',8:'--',9:'~',10:'!',11:'%',12:'^',13:'F',14:'H',15:'K',16:'U',17:'T'}

                #for i, label in enumerate(labels):

                 #   alpha = alphas[i]
                  #  if(label>=100):
                   #     line = str(t[i, 0]) + ' ' + str(t[i, 1]) + ' ' + str(label_to_mark[label-100]) + ' ' + str(alpha / 10)
                   # else:
                   #     line = str(t[i, 0]) + ' ' + str(t[i, 1]) + ' ' + str(label_to_mark[label]) + ' ' + str(alpha / 10)
                   # writer.write(line + '\n')
