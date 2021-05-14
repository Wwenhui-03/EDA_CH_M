# 该实验，验证不同的增广alpha


#user inputs

#size folders
sizes = ['1_tiny', '2_small', '3_standard', '4_full']
#sizes = ['4_full']
size_folders = ['size_data_f3/' + size for size in sizes]

#dataset folder
# datasets =[]
datasets =['hotel','tnews','fudan']

#number of output classes
#num_classes_list=[]
#categories = []
num_classes_list = [2,15,4]
categories = [['0', '1'],['100','101','102','103','104','106','107','108','109','110','112','113','114','115','116'],['19','31','34','39']]
#alpha values we care about
num_aug_list = [0]

#number of words for input
input_size_list = [50, 50, 50, 25, 25] 

#word2vec dictionary
huge_word2vec = 'word2vec/word2vec.txt'
word2vec_len = 300 # don't want to load the huge pickle every time, so just save the words that are actually used into a smaller dictionary
