from c_config import * 
from methods import *
from numpy.random import seed
import os
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 程序最多只能占用指定gpu50%的显存8
#config.gpu_options.allow_growth = True      #程序按需申请内存
#sess = tf.compat.v1.Session(config=config)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


seed(5)

###############################
#### run model and get acc ####
###############################

def run_cnn(train_file, test_file, num_classes, percent_dataset):

	#initialize model
	model = build_cnn(input_size, word2vec_len, num_classes)

	#load data
	train_x, train_y = get_x_y(train_file, num_classes, word2vec_len, input_size, word2vec, percent_dataset)
	test_x, test_y = get_x_y(test_file, num_classes, word2vec_len, input_size, word2vec, 1)

	#implement early stopping
	callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

	#train model
	model.fit(	train_x, 
				train_y, 
				epochs=1000000,
				callbacks=callbacks,
				validation_split=0.1, 
				batch_size=1,
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

###############################
############ main #############
###############################

if __name__ == "__main__":

	for see in range(5):

		seed(see)
		print('seed:', see)
		filename='outputs_f3/' + get_now_str()
		writer = open(filename+'seed'+str(see) + '.txt', 'a+')

		#for each size dataset
		for size_folder in size_folders:

			writer.write(size_folder + '\n')

			#get all six datasets
			dataset_folders = [size_folder + '/' + s for s in datasets]

			#for storing the performances
			performances = {num_aug:[] for num_aug in num_aug_list}

			#for each dataset
			for i in range(len(dataset_folders)):

				#initialize all the variables
				dataset_folder = dataset_folders[i]
				dataset = datasets[i]
				num_classes = num_classes_list[i]
				input_size = input_size_list[i]
				word2vec_pickle = dataset_folder + '/word2vec.p'
				word2vec = load_pickle(word2vec_pickle)

				#test each num_aug value
				for num_aug in num_aug_list:

					train_path = dataset_folder + '/train_' + str(num_aug) + '.txt'
					test_path = 'size_data_f3/test/' + dataset + '/test.txt'
					acc = run_cnn(train_path, test_path, num_classes, percent_dataset=1)
					performances[num_aug].append(acc)
					writer.write(train_path + ',' + str(acc))
			writer = open(filename + 'seed' + str(see) + '.txt', 'a+')
			writer.write(str(performances) + '\n')
			writer.close()
			writer = open(filename + 'seed' + str(see) + '.txt', 'a+')
			print()
			for num_aug in performances:
				line = str(num_aug) + ' : ' + str(sum(performances[num_aug])/len(performances[num_aug]))
				writer.write(line + '\n')
				print(line)
			print(performances)
			writer.close()
			writer = open(filename + 'seed' + str(see) + '.txt', 'a+')

		writer.close()
