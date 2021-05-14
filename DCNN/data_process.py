#coding:utf-8

import re
import os
from xml.dom import minidom
from urllib.parse import urlparse
import io
import sys
from methods import *
from sklearn.utils import shuffle

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')  # 改变标准输出的默认编码

def split():
    p = re.compile('</doc>',re.S)
    end = '</doc>'
    fileContent = open('./news_tensite_xml.dat','rb').read().decode('GBK',errors='ignore')  #读文件内容
    paraList = p.split(fileContent)     #根据</doc>对文本进行切片

    fileWriter = open('./files/0.txt','a',encoding='GBK')  #创建一个写文件的句柄
    #遍历切片后的文本列表
    for paraIndex in range(len(paraList)):
        #print(paraList[paraIndex])
        fileWriter.write(paraList[paraIndex])   #先将列表中第一个元素写入文件中
        if(paraIndex != len(paraList)):         #不加if这两行的运行结果是所有的</doc>都没有了，除了最后分割的文本
            fileWriter.write(end)
        if((paraIndex+1)%5000==0):              #5000个切片合成一个.txt文本
            fileWriter.close()
            fileWriter = open('./files/'+str((paraIndex+1)/5000)+'.txt','a',encoding='GBK'); #重新创建一个新的句柄，等待写入下一个切片元素。注意这里文件名的处理技巧。
    fileWriter.close()          #关闭最后创建的那个写文件句柄
    print('finished')




def file_fill():
    file_dir=   './files'
    for root, dirs, files in os.walk(file_dir):  # 扫描该目录下的文件夹和文件，返回根目录路径，文件夹列表，文件列表
        print(root)
        print(dirs)
        print(files)
        for f in files:
            tmp_dir = './sougou_after2' + '/' + f  # 加上标签后的文本
            text_init_dir = file_dir + '/' + f  # 原始文本
            print(text_init_dir)
            print(tmp_dir)
            file_source = open(text_init_dir, 'r', encoding='GBK')  # 打开文件，并将字符按照utf-8编码，返回unicode字节流
            print(file_source)
            ok_file = open(tmp_dir, 'a+', encoding='utf-8')
            start = '<docs>\n'
            end = '</docs>'
            line_content = file_source.readlines()  # 按行读取
            ok_file.write(start)
            for lines in line_content:
                text_temp = lines.replace('&', '.')  # 替换：replace(old,new,[max]) max最多替换的次数
                text = text_temp.replace('', '')
                ok_file.write(text)
            ok_file.write('\n' + end)

            file_source.close()
            ok_file.close()
    print('finished!')


def file_read():
    file_dir='./sougou_after2'
    # 建立url和类别的映射词典,可以参考搜狗实验室的对照.txt,有18类，这里增加了奥运，减少了社会、国内和国际新闻
    dicurl = {'auto.sohu.com': 'qiche', 'it.sohu.com': 'hulianwang', 'health.sohu.com': 'jiankang',
              'sports.sohu.com': 'tiyu',
              'travel.sohu.com': 'lvyou', 'learning.sohu.com': 'jiaoyu', 'career.sohu.com': 'zhaopin',
              'cul.sohu.com': 'wenhua',
              'mil.news.sohu.com': 'junshi', 'house.sohu.com': 'fangchan', 'yule.sohu.com': 'yule',
              'women.sohu.com': 'shishang',
              'media.sohu.com': 'chuanmei', 'gongyi.sohu.com': 'gongyi', '2008.sohu.com': 'aoyun',
              'business.sohu.com': 'shangye'}
    path = "./sougou_all/"
    for root, dirs, files in os.walk(file_dir):
        for f in files:
            print(f)
            doc = minidom.parse(file_dir + '/' + f)
            root = doc.documentElement
            claimtext = root.getElementsByTagName("content")
            claimurl = root.getElementsByTagName("url")
            for index in range(0, len(claimurl)):
                if (claimtext[index].firstChild == None):
                    continue
                url = urlparse(claimurl[index].firstChild.data)
                if url.hostname in dicurl:
                    if not os.path.exists(path + dicurl[url.hostname]):
                        os.makedirs(path + dicurl[url.hostname])
                    fp_in = open(
                        path + dicurl[url.hostname] + "/%d.txt" % (len(os.listdir(path + dicurl[url.hostname])) + 1),
                        "w")
                    temp_bytescontent = (claimtext[index].firstChild.data).encode('GBK',
                                                                                  'ignore')  # 这里的ignore是说，如果编码过程中有GBK不认识的字符可以忽略
                    fp_in.write(temp_bytescontent.decode('GBK', 'ignore'))
    print('finished!')

def orig_split():
    file_dir = "./sougou_all"
    sougou = open('sougou_all.txt', 'w', encoding='utf-8-sig')
    num = 0
    all = 0
    for root, dirs, files in os.walk(file_dir):  # 扫描该目录下的文件夹和文件，返回根目录路径，文件夹列表，文件列表
        print(root)
        print(dirs)
        print(files)
        for f in files:
            text_init_dir = root + '/' + f  # 原始文本
            file_source = open(text_init_dir, 'r')
            line_content = file_source.read()
            line_content = line_content.replace('\n', '')
            all = all + 1
            if (root == './sougou_all/junshi'):
                sougou.write('1\t' + line_content + '\n')
                num = num + 1
            else:
                sougou.write('0\t' + line_content + '\n')

            file_source.close()
    sougou.close()
    print(num)
    sougou = 'sougou_all.txt'
    percent_train = 0.7
    all_lines = open(sougou, 'r', encoding='utf-8-sig').readlines()
    all_lines = shuffle(all_lines)
    train_lines = all_lines[:int(percent_train * len(all_lines))]
    test_lines = all_lines[int(percent_train * len(all_lines)):]
    train = open('sougou/train.txt', 'w', encoding='utf-8-sig')
    test = open('sougou/test.txt', 'w', encoding='utf-8-sig')
    for line in train_lines:
        line = line.split('\t')
        if (len(line) > 1):
            train.write(line[0] + '\t' + line[1])
    train.close()
    for line in test_lines:
        line = line.split('\t')
        if (len(line) > 1):
            test.write(line[0] + '\t' + line[1])
    test.close()
def split_jun_other():
    file_dir = "./sougou_all"
    sougou_jun = open('sougou_jun.txt', 'w', encoding='utf-8-sig')
    sougou_other = open('sougou_other.txt', 'w', encoding='utf-8-sig')
    num = 0
    all = 0
    for root, dirs, files in os.walk(file_dir):  # 扫描该目录下的文件夹和文件，返回根目录路径，文件夹列表，文件列表
        print(root)
        print(dirs)
        print(files)
        for f in files:
            text_init_dir = root + '/' + f  # 原始文本
            file_source = open(text_init_dir, 'r')
            line_content = file_source.read()
            line_content = line_content.replace('\n', '')
            all = all + 1
            if (root == './sougou_all/junshi'):
                sougou_jun.write('1\t' + line_content + '\n')
                num = num + 1
            else:
                sougou_other.write('0\t' + line_content + '\n')
            file_source.close()
    sougou_jun.close()
    sougou_other.close()


def aug_split(alpha_sr=0.25, alpha_ri=0.25, alpha_rs=0.25, p_rd=0.1, num_aug=9):

    # pre-existing file locations
    train_orig = 'sougou_jun.txt'

    # file to be created
    train_aug_st = 'sougou_jun_aug'+str(num_aug)+'.txt'

    # augmentation
    gen_standard_aug(train_orig, train_aug_st, alpha_sr, alpha_ri, alpha_rs, p_rd, num_aug)

    sougou_jun = open('sougou_jun_aug'+str(num_aug)+'.txt', 'r', encoding='utf-8-sig').readlines()
    sougou_other=open('sougou_other.txt', 'r', encoding='utf-8-sig').readlines()
    sougou = open('sougou_all_aug'+str(num_aug)+'.txt', 'w', encoding='utf-8-sig')
    for line in sougou_jun:
        line = line.split('\t')
        if (len(line) > 1):
            sougou.write(line[0] + '\t' + line[1])
    for line in sougou_other:
        line = line.split('\t')
        if (len(line) > 1):
            sougou.write(line[0] + '\t' + line[1])
    sougou.close()

    percent_train = 0.7
    all_lines = open('sougou_all_aug'+str(num_aug)+'.txt', 'r', encoding='utf-8-sig').readlines()
    all_lines = shuffle(all_lines)
    train_lines = all_lines[:int(percent_train * len(all_lines))]
    test_lines = all_lines[int(percent_train * len(all_lines)):]
    train = open('sougou/train_aug'+str(num_aug)+'.txt', 'w', encoding='utf-8-sig')
    test = open('sougou/test_aug'+str(num_aug)+'.txt', 'w', encoding='utf-8-sig')
    for line in train_lines:
        line = line.split('\t')
        if (len(line) > 1):
            train.write(line[0] + '\t' + line[1])
    train.close()
    for line in test_lines:
        line = line.split('\t')
        if (len(line) > 1):
            test.write(line[0] + '\t' + line[1])
    test.close()

#if __name__ == "__main__":
    #split()
    #file_fill()
    #file_read()
    #orig_split()
    #split_jun_other()
