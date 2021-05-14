#user inputs

#load hyperparameters
sizes = ['1_tiny', '2_small','3_standard', '4_full']
size_folders = ['size_data_t1/' + size for size in sizes]

#datasets
#datasets = ['hotel', 'tnews']
datasets =['fudan']
#number of output classes
#num_classes_list = [2, 2, 4, 2]
num_classes_list=[4]
categories = [['19','31','34','39']]
#categories = [['0', '1'], ['0', '1'], ['0', '1', '2', '3'], ['0', '1']]  # weibo
#categories = [['0', '1'], ['100','101','102','103','104','106','107','108','109','110','112','113','114','115','116']]


#number of augmentations per original sentence
n_aug_list_dict = {'size_data_t1/1_tiny': [32, 32, 32, 32, 32], 
					'size_data_t1/2_small': [32, 32, 32, 32, 32],
					'size_data_t1/3_standard': [16, 16, 16, 16, 4],
					'size_data_t1/4_full': [16, 16, 16, 16, 4]}

#number of words for input
input_size_list = [150, 50]

#word2vec dictionary
huge_word2vec = 'word2vec/word2vec.txt'
word2vec_len = 300