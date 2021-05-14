#user inputs
#增广在不同大小的数据集上的表现
#size folders
sizes = ['1_tiny','2_small', '3_standard', '4_full']
size_folders = ['size_data_f1/' + size for size in sizes]

#augmentation methods'sr',  'rd',
a_methods = ['rs', 'ri']

#dataset folder

datasets = ['hotel','tnews','fudan']
#datasets = ['tnews','fudan']

#number of output classes
#num_classes_list = [2, 2, 4, 2]
num_classes_list = [2,15,4]
#categories = [['0', '1'], ['0', '1'], ['0', '1', '2', '3'], ['0', '1']]  # weibo
#categories = [['0', '1'],['100','101','102','103','104','106','107','108','109','110','112','113','114','115','116'],['19','31','34','39']]
categories = [['0', '1'],['0','1','2','3','4','5','7','8','9','10','11','12','13','14','6'],['0','1','2','3']]
#categories =[['100','101','102','103','104','106','107','108','109','110','112','113','114','115','116'],['19','31','34','39']]
#number of augmentations
n_aug_list_dict = {'size_data_f1/1_tiny': [16, 16, 16, 16, 16],
					'size_data_f1/2_small': [16, 16, 16, 16, 16],
					'size_data_f1/3_standard': [8, 8, 8, 8, 4],
					'size_data_f1/4_full': [8, 8, 8, 8, 4]}

#alpha values we care about
alphas = [0 , 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
#alphas = [0]
#number of words for input
input_size_list = [30, 30,150]

#word2vec dictionary
huge_word2vec = 'word2vec/word2vec.txt'
word2vec_len = 300 # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary
