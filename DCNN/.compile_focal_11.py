#coding=utf8
from models import *
import dataUtils
import numpy as np
import time
import os
import sklearn
from focal_loss import *
from time import gmtime, strftime
from data_process import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


class train():
    def __init__(self, x_train, x_dev, y_train, y_dev, x_test, y_test,num_aug):
        self.x_train = x_train
        self.x_dev = x_dev
        self.y_train = y_train
        self.y_dev = y_dev
        self.x_test = x_test
        self.y_test = y_test
        self.num_aug=num_aug


    def train(self, sent_length,num_class):
        embed_dim = 32
        ws = [8, 5]
        top_k = 4
        k1 = 19
        num_filters = [6, 14]
        batch_size = 50
        n_epochs = 30
        num_hidden = 100
        sentence_length = sent_length
        num_class = num_class
        evaluate_every = 10
        checkpoint_every = 10
        num_checkpoints = 5

        # --------------------------------------------------------------------------------------#
        def get_now_str():
            return str(strftime("%Y-%m-%d_%H-%M-%S", gmtime()))


        def init_weights(shape, name):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)

        sent = tf.placeholder(tf.int64, [None, sentence_length])
        y = tf.placeholder(tf.float64, [None, num_class])
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout")

        with tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.random_uniform([len(vocabulary), embed_dim], -1.0, 1.0), name="embed_W")
            sent_embed = tf.nn.embedding_lookup(W, sent)
            # input_x = tf.reshape(sent_embed, [batch_size, -1, embed_dim, 1])
            input_x = tf.expand_dims(sent_embed, -1)
            # [batch_size, sentence_length, embed_dim, 1]

        W1 = init_weights([ws[0], embed_dim, 1, num_filters[0]], "W1")
        b1 = tf.Variable(tf.constant(0.1, shape=[num_filters[0], embed_dim]), "b1")


        # 增加int()
        # W2 = init_weights([ws[1], int(embed_dim/2), num_filters[0], num_filters[1]], "W2")
        W2 = init_weights([ws[1], int(embed_dim), num_filters[0], num_filters[1]], "W2")
        b2 = tf.Variable(tf.constant(0.1, shape=[num_filters[1], embed_dim]), "b2")

        # 增加int
        # Wh = init_weights([int(top_k*embed_dim*num_filters[1]/4), num_hidden], "Wh")
        Wh = init_weights([int(top_k * embed_dim * num_filters[1] / 2), num_hidden], "Wh")
        bh = tf.Variable(tf.constant(0.1, shape=[num_hidden]), "bh")

        Wo = init_weights([num_hidden, num_class], "Wo")
        model = DCNN(batch_size, sentence_length, num_filters, embed_dim, top_k, k1)
        out = model.DCNN(input_x, W1, W2, b1, b2, k1, top_k, Wh, bh, Wo, dropout_keep_prob)



        # 损失函数/代替换
        with tf.name_scope("cost"):
            #y_=tf.nn.softmax(out)
            #print(out)
            cost =focal_loss(pred=tf.cast(out,tf.float64), y=tf.cast(y,tf.float64))
            #cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
        # train_step = tf.train.AdamOptimizer(lr).minimize(cost)



        with tf.name_scope("prediction"):
            prediction = tf.argmax(out, axis=1)
        	
        with tf.name_scope("label"):
            label = tf.argmax(y, axis=1)
        #cm = confusion_matrix(tf.argmax(y, axis = 1), tf.argmax(out, axis = 1))
        #fpr, tpr, _ = sklearn.metrics.roc_curve(tf.argmax(y, axis = 1)), tf.argmax(out, axis = 1))


        #with tf.name_scope("precision"):
        #    prec = cm[1][1]/(cm[1][1]+cm[0][1])

        #with tf.name_scope("recall"):
        #    recall = cm[1][1]/(cm[1][1]+cm[1][0])
        #with tf.name_scope("auc"):
        #    auc=auc(fpr, tpr)
        with tf.name_scope("accuracy"):
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(out, 1)), tf.float64))

        
        # -------------------------------------------------------------------------------------------#

        print('Started training')
        filename_train = 'result/focal_roc/'+str(self.num_aug) + '/aug' + '_' + get_now_str() +'train'+ '.txt'
        filename_vaild = 'result/focal_roc/'+str(self.num_aug)+ '/aug' + '_' + get_now_str() + 'vaild'+'.txt'

        with tf.Session() as sess:	
            
            # init = tf.global_variables_initializer().run()

            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 学习率函数
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cost)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cost)
            acc_summary = tf.summary.scalar("accuracy", acc)
            label_summary =  tf.summary.histogram("label", label)
            prediction_summary =  tf.summary.histogram("prediction", prediction)
            #prec_summary = tf.summary.scalar("precision", prec)
            #recall_summary = tf.summary.scalar("recall", recall)
            #auc_summary = tf.summary.scalar("auc", auc)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary,prediction_summary,label_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary, prediction_summary,label_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                feed_dict = {
                    sent: x_batch,
                    y: y_batch,
                    dropout_keep_prob: 0.5
                }
                out
                _, step, summaries, loss, accuracy ,predictions,labels= sess.run(
                    [train_op, global_step, train_summary_op, cost, acc, prediction, label],
                    feed_dict)
                labels_plus=np.insert(labels,0,values=1)
                predictions_plus=np.insert(predictions,0,values=1)
                cm = confusion_matrix(labels_plus, predictions_plus)
                fpr, tpr, _ = sklearn.metrics.roc_curve(labels_plus, predictions_plus)
                prec = cm[1][1]/(cm[1][1]+cm[0][1])
                recall = cm[1][1]/(cm[1][1]+cm[1][0])
                auc=sklearn.metrics.auc(fpr, tpr)
                #print(labels_plus,predictions_plus)
                print("TRAIN step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, auc {:g}".format(step, loss, accuracy,prec,recall,auc))
                train_summary_writer.add_summary(summaries, step)
                writerfile = open(filename_train, 'a+')
                writerfile.write(str(step)+','+str(loss)+','+str(accuracy)+','+str(prec)+','+str(recall)+','+str(auc) + '\n')
                writerfile.close()

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    sent: x_batch,
                    y: y_batch,
                    dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy ,predictions,labels = sess.run(
                    [global_step, dev_summary_op, cost, acc, prediction, label],
                    feed_dict)
                labels_plus=np.insert(labels,0,values=1)
                predictions_plus=np.insert(predictions,0,values=1)
                cm = confusion_matrix(labels_plus, predictions_plus)
                fpr, tpr, _ = sklearn.metrics.roc_curve(labels_plus, predictions_plus)
                prec = cm[1][1]/(cm[1][1]+cm[0][1])
                recall = cm[1][1]/(cm[1][1]+cm[1][0])
                auc=sklearn.metrics.auc(fpr, tpr)
                #print(labels_plus,predictions_plus)
                print("VALID step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}, auc {:g}".format(step, loss, accuracy,prec,recall,auc))
                writerfile = open(filename_vaild, 'a+')
                writerfile.write(str(step) + ',' + str(loss) + ',' + str(accuracy)+','+str(prec)+','+str(recall)+','+str(auc) + '\n')
                writerfile.close()

                if writer:
                    writer.add_summary(summaries, step)
                return accuracy, loss
            # 添加list强制装换
            batches = dataUtils.batch_iter(list(zip(self.x_train, self.y_train)), batch_size, n_epochs)

            # Training loop. For each batch...
            max_acc = 0
            best_at_step = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    acc_dev, _ = dev_step(self.x_dev, self.y_dev, writer=dev_summary_writer)
                    if acc_dev >= max_acc:
                        max_acc = acc_dev
                        best_at_step = current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("")
                if current_step % checkpoint_every == 0:
                    print('Best of valid = {}, at step {}'.format(max_acc, best_at_step))

            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            print('Finish training. On test set:')
            acc, loss = dev_step(self.x_test, self.y_test, writer=None)
            print(acc, loss)


if __name__ == "__main__":
    for num_aug in range(0,17):
        aug_split(num_aug=num_aug)
        train_path='./sougou/train_aug'+str(num_aug)+'.txt'
        test_path='./sougou/test_aug'+str(num_aug)+'.txt'
        dev_percent = 0.05
        # Load data
        print("Loading data...")
        x_, y_, vocabulary, vocabulary_inv,train_size, test_size,sent_length,num_class = dataUtils.load_data(train_path,test_path)

        x, x_test = x_[:-test_size], x_[-test_size:]
        y, y_test = y_[:-test_size], y_[-test_size:]
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        x_train, x_dev = x_shuffled[:int(-dev_percent*train_size)], x_shuffled[int(-dev_percent*train_size):]
        y_train, y_dev = y_shuffled[:int(-dev_percent*train_size)], y_shuffled[int(-dev_percent*train_size):]

        print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))

        yf = train(x_train, x_dev, y_train, y_dev, x_test, y_test,num_aug)
        yf.train(sent_length,num_class)
