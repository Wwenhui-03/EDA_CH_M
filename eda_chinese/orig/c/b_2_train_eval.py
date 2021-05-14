from b_config import * 
from methods import *
from numpy.random import seed
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu50%的显存8
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config=config)

seed(0)

###############################
#### run model and get acc ####
###############################

def run_model(train_file, test_file, num_classes, percent_dataset):

	#initialize model
	model = build_model(input_size, word2vec_len, num_classes)

	#load data
	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

	#implement early stopping
	callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

	#train model
	model.fit(	train_x, 
				train_y, 
				epochs=100000, 
				callbacks=callbacks,
				validation_split=0.1, 
				batch_size=128,
				shuffle=True, 
				verbose=0)
	#model.save('checkpoints/lol')
	#model = load_model('checkpoints/lol')

	#evaluate model
	y_pred = model.predict(test_x)
	test_y_cat = one_hot_to_categorical(test_y)
	y_pred_cat = one_hot_to_categorical(y_pred)
	acc = accuracy_score(test_y_cat, y_pred_cat)

	#clean memory???
	train_x, train_y = None, None
	gc.collect()

	#return the accuracy
	#print("data with shape:", train_x.shape, train_y.shape, 'train=', train_file, 'test=', test_file, 'with fraction', percent_dataset, 'had acc', acc)
	return acc

if __name__ == "__main__":

	#get the accuracy at each increment
	orig_accs = {dataset:{} for dataset in datasets}
	aug_accs = {dataset:{} for dataset in datasets}
	filename = 'outputs_f2/' + get_now_str()
	writer = open(filename + '.csv', 'a+')

	#for each dataset
	for i, dataset_folder in enumerate(dataset_folders):

		dataset = datasets[i]
		num_classes = num_classes_list[i]
		input_size = input_size_list[i]
		train_orig = dataset_folder + '/train_orig.txt'
		train_aug_st = dataset_folder + '/train_aug_st.txt'
		test_path = dataset_folder + '/test.txt'
		word2vec_pickle = dataset_folder + '/word2vec.p'
		word2vec = load_pickle(word2vec_pickle)

		for increment in increments:
			
			#calculate augmented accuracy
			aug_acc = run_model(train_aug_st, test_path, num_classes, increment)
			aug_accs[dataset][increment] = aug_acc

			#calculate original accuracy
			orig_acc = run_model(train_orig, test_path, num_classes, increment)
			orig_accs[dataset][increment] = orig_acc

			print(dataset, increment, orig_acc, aug_acc)
			writer.write(dataset + ',' + str(increment) + ',' + str(orig_acc) + ',' + str(aug_acc) + '\n')
			writer.close()
			writer = open(filename + '.csv', 'a+')
			gc.collect()

	print(orig_accs, aug_accs)
