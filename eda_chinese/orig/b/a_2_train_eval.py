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

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存8
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.compat.v1.Session(config=config)

###############################
#### run model and get acc ####
###############################
from run_rnn import *
###############################
############ main #############
###############################

if __name__ == "__main__":

	#for each method
	for a_method in a_methods:
		write_path='outputs_f1/'
		if not os.path.exists(write_path):
			os.makedirs(write_path)
		filename='outputs_f1/' + a_method + '_' + get_now_str() + '.txt'
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
				dataset_folder = dataset_folders[i]
				dataset = datasets[i]
				num_classes = num_classes_list[i]
				input_size = input_size_list[i]
				word2vec_pickle = dataset_folder + '/word2vec.p'
				word2vec = load_pickle(word2vec_pickle)
				category = categories[i]
				#test each alpha value
				for alpha in alphas:
					base_dir = dataset_folder
					train_path = dataset_folder + '/train_' + a_method + '_' + str(alpha) + '.txt'
					test_path = 'size_data_f1/test/' + dataset + '/test.txt'
					acc = run_rnn(train_path, test_path, num_classes, percent_dataset=1)
					#run_rnn(base_dir, filename, category)
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
