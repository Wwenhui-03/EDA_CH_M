import jieba
import synonyms
import random
from random import shuffle
jieba.load_userdict('userdict.txt')
random.seed(1)

########################################################################
# 停用词列表，使用哈工大停用词表
f = open('stopwords/HIT_stop_words.txt', encoding='UTF-8')
stop_words = list()
for stop_word in f.readlines():
    stop_words.append(stop_word[:-1])
########################################################################

########################################################################
# 文本清理
# cleaning up text
########################################################################
import re

def is_chinese(word):
    '''
    判断传入字符串是否为中文
    '''
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    match = zh_pattern.search(word)
    return match


def get_only_charactor(line):
    clean_line = ""

    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.replace("。", " ")
    line = line.replace(",", " ")
    line = line.replace("、", " ")
    line = line.replace("；", " ")
    line = line.replace("：", " ")
    line = line.replace("“", " ")
    line = line.replace("”", " ")
    line = line.replace("‘", " ")
    line = line.replace("’", " ")
    line = line.replace("！", " ")
    line = line.replace("？", " ")

    for char in line:
        if is_chinese(char):
            clean_line += char
        else:
            if char in 'qwertyuiopasdfghjklzxcvbnm':
                clean_line += '#CHAR#'
            if char in '1234567890':
                clean_line += '#NUM#'

    clean_line = re.sub('(#NUM#)+', '#NUM#', clean_line)
    clean_line = re.sub('(#CHAR#)+', '#CHAR#', clean_line)
    clean_line = re.sub(' +', ' ', clean_line)  # 删除多余空格

    if (len(clean_line) > 0):
        if clean_line[0] == ' ':
            clean_line = clean_line[1:]
    return clean_line


########################################################################
# 同义词替换
# 替换一个语句中的n个单词为其同义词
########################################################################
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    return synonyms.nearby(word)[0]


########################################################################
# 随机插入
# 随机在语句中插入n个词
########################################################################
def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    if(len(new_words)>1):
        while len(synonyms) < 1:

            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms)  ########
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)


########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    if (len(new_words) > 1):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


########################################################################
# 随机删除
# 以概率p删除语句中的词
########################################################################
def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


########################################################################
# EDA函数
def eda(sentence, alpha_sr=0.3, alpha_ri=0.2, alpha_rs=0.1, p_rd=0.15, num_aug=9):
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    # 区别于英文直接用空格划分，中文首先分词，再用空格隔开每个词，最后处理成英文相同格式
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_charactor(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    augmented_sentences.append(seg_list)

    return augmented_sentences


def SR(sentence, alpha_sr, n_aug=9):
    sentence = get_only_charactor(sentence)
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    #print(words)
    num_words = len(words)

    augmented_sentences = []
    n_sr = max(1, int(alpha_sr * num_words))

    for _ in range(n_aug):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_charactor(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    augmented_sentences.append(sentence)

    return augmented_sentences


def RI(sentence, alpha_ri, n_aug=9):
    sentence = get_only_charactor(sentence)
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    n_ri = max(1, int(alpha_ri * num_words))

    for _ in range(n_aug):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_charactor(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    augmented_sentences.append(sentence)

    return augmented_sentences


def RS(sentence, alpha_rs, n_aug=9):
    sentence = get_only_charactor(sentence)
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    n_rs = max(1, int(alpha_rs * num_words))

    for _ in range(n_aug):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_charactor(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    augmented_sentences.append(sentence)

    return augmented_sentences


def RD(sentence, alpha_rd, n_aug=9):
    sentence = get_only_charactor(sentence)
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []

    for _ in range(n_aug):
        a_words = random_deletion(words, alpha_rd)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_charactor(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

    augmented_sentences.append(sentence)

    return augmented_sentences
