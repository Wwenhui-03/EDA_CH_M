#user inputs

#dataset folder
#datasets = ['hotel','tnews']
datasets = ['fudan']
dataset_folders = ['increment_datasets_f2/' + dataset for dataset in datasets] 

#number of output classes
#num_classes_list = [2,15,4]
num_classes_list=[4]
categories = [['19','31','34','39']]
#categories = [['0', '1'],['100','101','102','103','104','106','107','108','109','110','112','113','114','115','116'],['19','31','34','39']]

#dataset increments
increments = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

#number of words for input
input_size_list = [150, 50]

#word2vec dictionary
huge_word2vec = 'word2vec/word2vec.txt'
word2vec_len = 300