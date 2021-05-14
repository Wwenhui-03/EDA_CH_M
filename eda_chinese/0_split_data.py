import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os


datasets=['tnews','hotel','fudan']
#datasets=['hotel']


for dataset in datasets:

    # a
    sizes = ['1_tiny', '2_small', '3_standard', '4_full','test']
    size_folders = ['size_data_f1/' + size for size in sizes]
    for size_folder in size_folders:
        write_path = size_folder + '/'+dataset+'/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)

    train_txt='raw_data/'+dataset+'.txt'
    percent_train=0.8
    percent_test=0.9
    all_lines = open(train_txt, 'r',encoding='utf-8-sig').readlines()
    all_lines=shuffle(all_lines)

    train_lines_full = all_lines[:int(percent_train * len(all_lines))]
    f_train_full = open('size_data_f1/4_full/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_full:
        line = line.split('\t')
        if (len(line)>1):
            f_train_full.write(line[0] + '\t' + line[1]  )
    f_train_full.close()

    train_lines_tiny = all_lines[:500]
    f_train_tiny = open('size_data_f1/1_tiny/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_tiny:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_tiny.write(line[0] + '\t' + line[1]  )
    f_train_tiny.close()

    train_lines_small = all_lines[:2000]
    f_train_small = open('size_data_f1/2_small/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_small:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_small.write(line[0] + '\t' + line[1]  )

    f_train_small.close()

    train_lines_standard = all_lines[:5000]
    f_train_standard = open('size_data_f1/3_standard/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_standard:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_standard.write(line[0] + '\t' + line[1]  )
    f_train_standard.close()

    test_lines = all_lines[int(percent_train * len(all_lines)):int((percent_test ) * len(all_lines))]
    f_test = open('size_data_f1/test/' + dataset + '/test.txt', 'w', encoding='utf-8-sig')
    for line in test_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_test.write(line[0] + '\t' + line[1]  )
    f_test.close()

    val_lines=all_lines[int((percent_test ) * len(all_lines)):]
    f_val = open('size_data_f1/test/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    #b
    write_path='increment_datasets_f2/'+dataset+'/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    train_lines_full = all_lines[:int(percent_train * len(all_lines))]
    f_train_full = open(write_path+'/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_full:
        line = line.split('\t')
        if (len(line) > 1):

            f_train_full.write(line[0] + '\t' + line[1]  )
    f_train_full.close()

    f_val = open(write_path+'/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:

         line = line.split('\t')
         if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    f_test = open(write_path+ '/test.txt', 'w', encoding='utf-8-sig')
    for line in test_lines:
        line = line.split('\t')
        if (len(line) > 1):

            f_test.write(line[0] + '\t' + line[1]  )
    f_test.close()


    # c
    sizes = ['1_tiny', '2_small', '3_standard', '4_full','test']
    size_folders = ['size_data_f3/' + size for size in sizes]
    for size_folder in size_folders:
        write_path = size_folder + '/'+dataset+'/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)

    train_txt='raw_data/'+dataset+'.txt'
    percent_train=0.8
    percent_test=0.9
    all_lines = open(train_txt, 'r',encoding='utf-8-sig').readlines()
    all_lines=shuffle(all_lines)

    train_lines_full = all_lines[:int(percent_train * len(all_lines))]
    f_train_full = open('size_data_f3/4_full/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_full:
        line = line.split('\t')
        if (len(line) > 1):

            f_train_full.write(line[0] + '\t' + line[1]  )
    f_train_full.close()

    train_lines_tiny = all_lines[:500]
    f_train_tiny = open('size_data_f3/1_tiny/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_tiny:
        line = line.split('\t')
        if (len(line) > 1):

            f_train_tiny.write(line[0] + '\t' + line[1]  )
    f_train_tiny.close()

    train_lines_small = all_lines[:2000]
    f_train_small = open('size_data_f3/2_small/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_small:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_small.write(line[0] + '\t' + line[1]  )

    f_train_small.close()

    train_lines_standard = all_lines[:5000]
    f_train_standard = open('size_data_f3/3_standard/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_standard:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_standard.write(line[0] + '\t' + line[1]  )
    f_train_standard.close()

    test_lines = all_lines[int(percent_train * len(all_lines)):int((percent_test ) * len(all_lines))]
    f_test = open('size_data_f3/test/' + dataset + '/test.txt', 'w', encoding='utf-8-sig')
    for line in test_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_test.write(line[0] + '\t' + line[1]  )
    f_test.close()

    val_lines=all_lines[int((percent_test ) * len(all_lines)):]
    f_val = open('size_data_f3/1_tiny/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    f_val = open('size_data_f3/2_small/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    f_val = open('size_data_f3/3_standard/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    f_val = open('size_data_f3/4_full/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1] )
    f_val.close()

    # d
    write_path='special_f4/' + dataset + '/'
    if not os.path.exists(write_path):
            os.makedirs(write_path)
    test_lines = all_lines[:100]
    f_test = open('special_f4/' + dataset + '/test.txt', 'w', encoding='utf-8-sig')
    for line in test_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_test.write(line[0] + '\t' + line[1]  )
    f_test.close()

    # e
    sizes = ['1_tiny', '2_small', '3_standard', '4_full', 'test']
    size_folders = ['size_data_t1/' + size for size in sizes]
    for size_folder in size_folders:
        write_path = size_folder + '/' + dataset + '/'
        if not os.path.exists(write_path):
            os.makedirs(write_path)

    train_lines_full = all_lines[:int(percent_train * len(all_lines))]
    f_train_full = open('size_data_t1/4_full/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_full:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_full.write(line[0] + '\t' + line[1]  )
    f_train_full.close()

    train_lines_tiny = all_lines[:500]
    f_train_tiny = open('size_data_t1/1_tiny/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_tiny:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_tiny.write(line[0] + '\t' + line[1]  )
    f_train_tiny.close()

    train_lines_small = all_lines[:2000]
    f_train_small = open('size_data_t1/2_small/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_small:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_small.write(line[0] + '\t' + line[1]  )

    f_train_small.close()

    train_lines_standard = all_lines[:5000]
    f_train_standard = open('size_data_t1/3_standard/' + dataset + '/train_orig.txt', 'w', encoding='utf-8-sig')
    for line in train_lines_standard:
        line = line.split('\t')
        if (len(line) > 1):
            f_train_standard.write(line[0] + '\t' + line[1]  )
    f_train_standard.close()

    test_lines = all_lines[int(percent_train * len(all_lines)):int((percent_test) * len(all_lines))]
    f_test = open('size_data_t1/test/' + dataset + '/test.txt', 'w', encoding='utf-8-sig')
    for line in test_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_test.write(line[0] + '\t' + line[1]  )
    f_test.close()

    val_lines=all_lines[int((percent_test ) * len(all_lines)):]
    f_val = open('size_data_t1/1_tiny/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')

    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    f_val = open('size_data_t1/2_small/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    f_val = open('size_data_t1/3_standard/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1]  )
    f_val.close()

    f_val = open('size_data_t1/4_full/' + dataset + '/val.txt', 'w', encoding='utf-8-sig')
    for line in val_lines:
        line = line.split('\t')
        if (len(line) > 1):
            f_val.write(line[0] + '\t' + line[1] )
    f_val.close()
