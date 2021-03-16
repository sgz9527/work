# coding=utf-8

"""
@File  : intro_block_ner.py
@Author: Xu Qiqiang
@Date  : 2021/1/27 0027
本模块为简历块内NER的API，主要包括基本信息、教育背景、工作经历提取、荣誉证书提取、论文/专利提取、技术栈提取
"""
import re
import random
from special_string import *
import spacy
import pandas as pd
import jieba
import gensim
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Dropout, BatchNormalization, Input, LSTM, Dense
from tensorflow.keras import Model
from load_resume_data import stopwordslist
from crf import CRF
np.set_printoptions(suppress=True)
spacy.prefer_gpu()

# nlp = spacy.load('en_core_web_sm')  # Write a function to display basic entity info:
nlp = spacy.load('zh_core_web_lg')

one_hot = {'O': 0, 'S': 1, 'J-B': 2, 'J-I': 3}


def corpus_processing(file_path):

    with open(file_path, 'r',encoding='utf-8') as f:
        text = f.read()
        texts = text.split('********************************************************************')
        for i, t in enumerate(texts):
            texts[i] = jieba.lcut(re.sub(DEL_SPECIAL_PAT1, '', t))

    with open(file_path,'w', encoding='utf-8') as f:
        for words in texts:
            for word in words:
                if word != '\n':
                    f.write(word)
                    f.write('\n')
            f.write('###\n')


def seg(text, split):
    i=0
    while i < len(text) and text[i] not in split:
        i += 1
    if i+1 < len(text):
        return text[:i+1], text[i+1:]
    return text, ''

# -----------------------------------------------------基本信息---------------------------------------- #


def extract_item_base_info(texts, nlp, dic):
    """
    提取基本信息中的内容项：姓名、性别、年龄、工作时长、住址、籍贯、出生年月、联系方式、毕业院校、学历、政治面貌、职业方向、个人链接
    :param texts:句子列表
    :param nlp:
    :return:
    """
    for i, text in enumerate(texts):
        doc = nlp(text)
        # show_ents(doc)
        # print('-------------------------------------------------')
        if doc.ents:
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    res = re.search(CORRECT_NAME, ent.text)
                    if res is not None:
                        # print(res)
                        continue
                    dic['name'] .append(ent.text)
                elif ent.label_ == 'ORG':  # 可能是学校名、公司名、网址形式（需要修正）
                    res = re.search(CORRECT_LINK, ent.text)
                    res1 = re.search(CORRECT_LINK1, ent.text)
                    if res is not None or res1 is not None:
                        pass
                    else:
                        dic['school'].append(ent.text)
                elif ent.label_ == 'DATE' or ent.label_ == 'CARDINAL':  # 可能是长整型时间字符串（造成数字类型误判为日期）、规则化时间字符串
                    if len(ent.text) in [6, 7, 8, 11, 12, 13, 14] and re.search(r'[^\d\-+]', ent.text) is None:
                        # 如果数字长度符合该范围以及仅仅包含指定数字和字符，则为联系方式
                        # dic['contact'].append(ent.text)
                        pass
                    elif re.search('年[ ]{0,6}龄|岁|手[ ]{0,6}机|联[ ]{0,6}系[ ]{0,6}方[ ]{0,6}式|'
                                   '电[ ]{0,6}话[ ]{0,6}号[ ]{0,6}码|号[ ]{0,6}码|'
                                   '电[ ]{0,6}话|周[ ]{0,6}岁|岁[ ]{0,6}数|Q[ ]{0,6}Q|微[ ]{0,6}信', text) is not None:
                        # item = re.sub(':|：|年龄|岁|岁数', '', ent.text)
                        # dic['age'].append(item)
                        pass
                    elif i > 0:
                        if re.search('手[ ]{0,10}机|联[ ]{0,10}系[ ]{0,10}方[ ]{0,10}式|'
                                     '电[ ]{0,10}话[ ]{0,10}号[ ]{0,10}码|号[ ]{0,10}码|电[ ]{0,10}话|年[ ]{0,10}龄|'
                                     '岁[ ]{0,10}数|QQ|微[ ]{0,10}信', texts[i-1], re.IGNORECASE) is not None:
                            # dic['contact'].append(ent.text)
                            pass
                        elif re.search('年龄|岁数', texts[i-1]) is not None:
                            # item = re.sub(':|：|年龄|岁|岁数', '', ent.text)
                            # dic['age'].append(item)
                            pass
                        else:
                            dic['work_time'] .append(ent.text)
                    else:
                        dic['work_time'] .append(ent.text)

                elif ent.label_ == 'GPE':  # 可能是住址、籍贯。用规则加以区分：如果有前缀提示，往前搜索字符“地址，住址...”、“籍贯、户口、国籍等”
                    if i == 0:
                        res = re.search('住[ ]{0,10}址|地[ ]{0,10}址|现[ ]{0,10}居[ ]{0,10}住[ ]{0,10}'
                                        '地|居[ ]{0,10}住[ ]{0,10}地|居[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '所[ ]{0,10}在[ ]{0,10}地|工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位|'
                                        '工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址|'
                                        '常[ ]{0,10}居[ ]{0,10}住[ ]{0,10}地|常[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址', text)
                        if res is not None:
                            dic['place'].append(ent.text)
                        else:
                            dic['city'].append(ent.text)
                    else:
                        res = re.search('住[ ]{0,10}址|地[ ]{0,10}址|现[ ]{0,10}居[ ]{0,10}住[ ]{0,10}'
                                        '地|居[ ]{0,10}住[ ]{0,10}地|居[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '所[ ]{0,10}在[ ]{0,10}地|工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位|'
                                        '工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址|'
                                        '常[ ]{0,10}居[ ]{0,10}住[ ]{0,10}地|常[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址', text)

                        res1 = re.search('住[ ]{0,10}址|地[ ]{0,10}址|现[ ]{0,10}居[ ]{0,10}住[ ]{0,10}'
                                        '地|居[ ]{0,10}住[ ]{0,10}地|居[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '所[ ]{0,10}在[ ]{0,10}地|工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位|'
                                        '工[ ]{0,10}作[ ]{0,10}单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址|'
                                        '常[ ]{0,10}居[ ]{0,10}住[ ]{0,10}地|常[ ]{0,10}住[ ]{0,10}地[ ]{0,10}址|'
                                        '单[ ]{0,10}位[ ]{0,10}地[ ]{0,10}址', texts[i-1])
                        if res is not None or res1 is not None:
                            dic['place'].append(ent.text)
                        else:
                            dic['city'].append(ent.text)
                else:
                    pass


def extract_item_by_rule_base_info(texts, dic):
    """
    提取 微信、QQ、手机号码、邮箱、性别、政治面貌、学历、年龄、个人链接
    :param texts:
    :return:
    """
    for i, text in enumerate(texts):
        # 检测邮箱
        res = re.search(r"[-_\w\.]{0,64}[ ]{0,5}@[ ]{0,5}([-\w]{1,63}\.)*[-\w]{1,63}", text)
        if res is not None:
            # print('email:', res.group())
            dic['contact'].append('邮箱:'+res.group())
        # 检测手机号
        res = re.search('[\d]{0,5}[+\-]?[\d]{7,11}', text)
        if res is not None:
            res1 = re.search('微[ ]{0,10}信|Q[ ]{0,10}Q|email', text, re.IGNORECASE)
            res2 = re.search('手[ ]{0,10}机|电[ ]{0,10}话|座[ ]{0,10}机|号[ ]{0,10}码', text)
            if res1 is None or res2 is not None:
                text1 = res.group()
                if re.search('[\-+]', text1) is not None:
                    dic['contact'].append('电话号码:' + res.group())
                else:
                    if len(text1) == 11:
                        dic['contact'].append('电话号码:' + res.group())
        # 检测微信
        res = re.search(r'(微[ ]{0,10}信[^a-zA-Z\d@#^&*%$!_-]*)[:：\- ]{0,2}([a-zA-Z\d@#^&*%$!_-]{3,})', text)
        if res is not None:
            # print(res.groups())
            # print('微信：', res.group())
            dic['contact'].append('微信:'+res.group(2))
        else:
            res = re.search(r'([a-zA-Z\d@#^&*%$!_-]{3,})([(（【{\[\-]微[ ]{0,10}信[)}）】\]]*)', text)
            if res is not None:
                # print('微信:', res.group(1))
                dic['contact'].append('微信:'+res.group(1))
        # 检测性别
        res = re.search('(性[ ]{0,10}别)?[:：\- ]?([男女]).*', text)
        if res is not None:
            # print('性别:', res.group(2))
            dic['gender'].append(res.group(2))
        # 检测政治面貌
        res = re.search('共[ ]{0,5}青[ ]{0,5}团[ ]{0,5}员|群[ ]{0,5}众|党[ ]{0,5}员|'
                        '中[ ]{0,5}共[ ]{0,5}党[ ]{0,5}员|普[ ]{0,5}通[ ]{0,5}公[ ]{0,5}民', text)
        if res is not None:
            # print('政治面貌：', res.group())
            dic['politic'].append(res.group())
        # 检测学历
        res = re.search(r'(学[ ]{0,10}位|学[ ]{0,10}历|学[ ]{0,10}历\\学[ ]{0,10}位)?'
                        r'[:：\-]?(小[ ]{0,10}学|初[ ]{0,10}中|高[ ]{0,10}中|专[ ]{0,10}科|本[ ]{0,10}'
                        r'科|学[ ]{0,10}士|研[ ]{0,10}究[ ]{0,10}生|硕[ ]{0,10}士|博[ ]{0,10}士)', text)
        if res is not None:
            # print('学历:', res.group(2))
            dic['scholar'].append(res.group(2))
        # 年龄
        res = re.search(r'(年[ ]{0,10}龄)[:\-： ]{0,3}([\d]{1,2})', text)
        if res is not None:
            # print(res.groups())
            # print(res.string)
            # print('年龄:', res.group(2))
            dic['age'].append(res.group(2))
        else:
            res = re.search(r'([\d]{1,2})(周[ ]{0,10}岁|岁)', text)
            if res is not None:
                # print(res.string)
                # print('年龄:', res.group(1))
                dic['age'].append(res.group(1))
        res = re.search(CORRECT_LINK, text)
        res1 = re.search(CORRECT_LINK1, text)
        if res is not None:
            print(res.groups())
            dic['link'].append(res.group())
        elif res1 is not None:
            print(res.groups())
            dic['link'].append(res1.group())


# -----------------------------------------------------基本信息(job extraction)---------------------------------------- #


def load_job(file_path):
    job_info = pd.DataFrame(columns=['job_direction', 'job_list'])
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = re.sub(r'\n', ' ', text)
    text_list = text.split('####')
    # print(text_list)
    for text in text_list:
        text = text.split(' ')
        if text[0] == '' or text[0] == ' ':
            job_info = job_info.append({'job_direction': text[1], 'job_list': text[2:]}, ignore_index=True)
        else:
            job_info = job_info.append({'job_direction': text[0], 'job_list': text[1:]}, ignore_index=True)
    # print(job_info)
    return job_info


def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - ' + str(ent.start_char) + ' - ' + str(ent.end_char) + ' - ' + ent.label_ + ' - ' +
                  str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')


def freq_analyze(job_info):
    dictionary = {}  # 装载所有的job单词
    word_list = []
    stop_words = stopwordslist()
    stop_words.append(' ')
    jieba.load_userdict(USER_DIC_PATH)
    for i in range(job_info.shape[0]):
        text = ' '.join(job_info.iloc[i, 1])
        text = re.sub(DEL_SPECIAL_PAT, '', text)
        text_list = jieba.lcut(text)
        text_list = [text for text in text_list if text != ' ']
        job_info.iloc[i, 1] = text_list
    count = 0
    for i in range(job_info.shape[0]):
        text_list = job_info.iloc[i, 1]
        for word in text_list:
            if word in dictionary.keys():
                dictionary[word][1] += 1  # 第二项为频率，第一项为索引
            else:
                word_list.append(word)
                dictionary[word] = [count, 1]
                count += 1
    bitmap = [[] for i in range(1000)]
    freq = []
    for key in dictionary.keys():
        bitmap[dictionary[key][1]].append(key)
    for i in range(len(bitmap) - 1, -1, -1):
        if bitmap[i]:
            freq.append([bitmap[i], i])
    print(dictionary)
    print(word_list)
    print('--------------------------------------------------------------')
    for f in freq:
        print(f)


def create_job_sentence(job_list, sentences, labels):
    """
    标签分为4种：O, S, J-B, J-I
    :param job_list: [job1, job2, ..., job_n]
    :param sentences: [ [word1, word2,...,word_m],  [word1, word2,...,word_m], ..., [word1, word2,...,word_m] ]
    :param labels: [ [l1, l2, ..., lm], [l1, l2, ..., lm], ... , [l1, l2, ..., lm] ]
    :return:
    """
    key = ['应聘职位', '应聘', '意向', '岗位', '意向岗位为', '意向岗位', '意向为', '应聘职位为',
           '担任过', '担任', '任职', '目标岗位', '目标岗位为', '职位', '职位目标是', '职业方向是', '曾担任过',
           '职位是', '岗位是', '想找', '工作目标岗位', '工作岗位', '工作', '工作是', '工作岗位为', '工作岗位是', '求职经历', '求职目标',
           '实习过', '做过', '想做', '胜任', '准备找', '准备寻求', '寻求', '准备', '/', '/', '/', '', '', '', '', '', '', '']
    characters = [' ', '-', '——', ':', '：', '', '']
    describe = ['分享', '介绍', '推荐', '建议', '发现', '觉得', '有过', '担任过', '经历过', '担任', '一年', '两年', '三年的', '四年', '五年的', '六年',
                '七年', '八年的', '九年', '十年的', '一个月', '两个月', '三个月', '半年', '四个月的', '五个月的', '六个月', '七个月的', '八个月', '九个月的'
        , '十个月的', '11个月的', '1年', '2年', '3年', '4年', '5年', '6年的', '7年的', '8年', '9年', '10年','公司',
                '企业', '招聘', '/', '/', '', '', '', '', '', '', '', '', '', '']
    describe2 = ['工作经验', '的工作', '很不错', '是一个很好的职位', '这个职位', '的实习经验', '实习经历', '求职经历', '面试经历', '经验',
                 '经历', '工作介绍', '工作经历', '实习经验', '一年', '部门', '方向', '岗位', '相关', '内容', '领域', '工作期间', '前景很好', '待遇', '薪资待遇',
                 '岗位描述', '岗位职责', '的职责', '的薪资待遇', '的内容', '的技术栈', '的待遇', '/', '/', '/', '', '', '', '', '', '', '', '', '']

    # template1: key[i] + characters[i] + job_list[i] 800条
    # template2: describe[i]+ job_list[i] + describe2[i] 800条
    # template3 job_list[i] 500条
    # template4 no job 500条
    jieba.load_userdict(USER_DIC_PATH)
    for i in range(800):
        sen = []
        label = []
        d1 = random.choice(key)
        d1 = jieba.lcut(d1)
        for d in d1:
            sen.append(d)
            label.append('O')
        sen.append(random.choice(characters))
        label.append('O')
        job = random.choice(job_list)
        job = jieba.lcut(job)
        if len(job) == 1:
            sen.append(job[0])
            label.append('S')
        else:
            for j, word in enumerate(job):
                sen.append(word)
                if j == 0:
                    label.append('J-B')
                else:
                    label.append('J-I')
        if len(sen)!=0:
            sentences.append(sen)
            labels.append(label)

    for i in range(800):
        sen = []
        label = []
        d1 = random.choice(describe)
        job = random.choice(job_list)
        d2 = random.choice(describe2)
        d1 = jieba.lcut(d1)
        for d in d1:
            sen.append(d)
            label.append('O')
        job = jieba.lcut(job)
        if len(job) == 1:
            sen.append(job[0])
            label.append('S')
        else:
            for j, word in enumerate(job):
                sen.append(word)
                if j == 0:
                    label.append('J-B')
                else:
                    label.append('J-I')
        d2 = jieba.lcut(d2)
        for d in d2:
            sen.append(d)
            label.append('O')
        if len(sen)!=0:
            sentences.append(sen)
            labels.append(label)

    for i in range(500):
        sen = []
        label = []
        job = random.choice(job_list)
        job = jieba.lcut(job)
        if len(job) == 1:
            sen.append(job[0])
            label.append('S')
        else:
            for j, word in enumerate(job):
                sen.append(word)
                if j == 0:
                    label.append('J-B')
                else:
                    label.append('J-I')
        if len(sen)!=0:
            sentences.append(sen)
            labels.append(label)

    # for i, sentence in enumerate(sentences):
        # print(sentence)
    #     print(labels[i])
    #     print('--')


def one_hot_label(label):
    num = one_hot[label]
    res = []
    for i in range(4):
        if i == num:
            res.append(1)
        else:
            res.append(0)
    return res


def word2feature(word2vec, sentences, labels):
    """
    转为vector
    :param word2vec:
    :param sentences:
    :param labels:
    :return:
    """
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            try:
                sentences[i][j] = word2vec[word]
            except KeyError:
                print('word2vec keyError:', word)
                # print(sentence)
                sentences[i][j] = word2vec['UAD']  # 用特殊字符代替未出现的词
                # print(sentences[i][j])
                labels[i][j] = one_hot_label('O')
                continue
            labels[i][j] = one_hot_label(labels[i][j])


def txt2feature_predict(word2vec, texts):
    return word2feature_predict(word2vec, texts)


def get_tokenize(texts):
    """

    :param texts: 未分词的文字
    :return: 分词后的文字
    """
    import copy
    tokenizes = copy.deepcopy(texts)
    jieba.load_userdict(USER_DIC_PATH)
    for i, text in enumerate(tokenizes):
        text = re.sub(DEL_SPECIAL_PAT2, '', text)
        tokenizes[i] = jieba.lcut(text)
    return tokenizes


def word2feature_predict(word2vec, tokenizes):
    """

    :param word2vec:
    :param tokenizes: 文字
    :return: 向量
    """
    import copy
    vec_arr = copy.deepcopy(tokenizes)
    for i, tokenize in enumerate(vec_arr):
        for j, word in enumerate(tokenize):
            try:
                vec_arr[i][j] = word2vec[word]
            except KeyError:
                print('word2vec keyError:', word)
                # sentences[i][j] = np.array([0 for i in range(100)], dtype=np.float)
                vec_arr[i][j] = word2vec['UAD']  # 用特殊字符代替未出现的词
    return vec_arr


def get_entity(pre_res, tokenizes):
    """

    :param pre_res: [[0,1,2,2,3,0],[...],[...]]
    :param tokenizes: [[word1, word2, ...],[word1, word2, ...],[word1, word2,...]]
    :return:
    """
    entities = []
    for i, pre in enumerate(pre_res):
        tokenize = tokenizes[i]
        # print(tokenize)
        # print(pre)
        ent = []
        j = 0
        length = min(len(tokenize), 8)
        while j < length:
            if pre[j] == 1:  # s
                ent.append(tokenize[j])
                j += 1
            elif pre[j] == 2:
                str1 = tokenize[j]
                j += 1
                while j < length and pre[j] == 3:
                    str1 += tokenize[j]
                    j += 1
                ent.append(str1)
            else:
                j += 1
        entities.append(ent)
    return entities


def create_model():
    """
    LSTM+CRF
    :return:
    """
    input_tensor = Input(shape=(8, 100), name='input')
    x1 = Bidirectional(LSTM(units=100, activation='relu', return_sequences=True))(input_tensor)
    x1 = BatchNormalization()(x1)
    x2 = Bidirectional(LSTM(units=100, activation='relu', return_sequences=True))(x1)
    x2 = BatchNormalization()(x2)
    x2 = Dense(units=64, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)
    logit = Dense(units=4)(x2)
    crf = CRF(units=4, name='crf_layer')
    output = crf(logit)
    return Model(inputs=input_tensor, outputs=output), crf


# -----------------------------------------------------教育背景---------------------------------------- #
def extract_item_edu(texts, nlp, job_model, dic, word2vec):
    """
    提取教育背景中的内容项：学校、专业、学历、就读时间
    学校名称、专业名称：ner名称提取
    学历、时间：规则提取
    :param texts:句子列表
    :param nlp:
    :param job_model:识别专业
    :param word2vec:txt2vec模型
    :return:
    """
    for i, text in enumerate(texts):
        doc = nlp(text)
        # show_ents(doc)
        # print('-------------------------------------------------')
        if doc.ents:
            for ent in doc.ents:
                if ent.label_ == 'ORG':  # 可能是学校名、公司名
                    dic['school'].append(ent.text)
                else:
                    pass
    tokenizes = get_tokenize(texts)
    x = word2feature_predict(word2vec, tokenizes)
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=8, padding='post', dtype='float',
                                                              truncating='post')
    x = np.array(x)
    pre_y = job_model.predict(x)
    res = get_entity(pre_y, tokenizes)
    for jobs in res:
        for job in jobs:
            dic['major'].append(job)


def extract_item_edu_by_rule(texts, dic):
    """
    通过规则提取学历以及就读时间
    :param texts:
    :param dic:
    :return:None
    """
    for i, text in enumerate(texts):
        res = re.search(r"(\d{4}[ ]{0,3}/[ ]{0,3}\d{1,2}[ ]{0,3}"
                        r"/?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"
                        r"/[ ]{0,5}\d{1,2}[ ]{0,3}/?[ ]{0,3}\d{0,2}[ ]{0,3})?", text)
        if res is None:
            res = re.search(r"(\d{4}[ ]{0,3}\.[ ]{0,3}\d{1,2}[ ]{0,3}"
                        r"\.?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"
                        r"\.[ ]{0,5}\d{1,2}[ ]{0,3}\.?[ ]{0,3}\d{0,2}[ ]{0,3})?", text)
            if res is None:
                res = re.search(r"(\d{4}[ ]{0,3}-[ ]{0,3}\d{1,2}[ ]{0,3}"
                        r"-?[ ]{0,3}\d{0,2}[ ]{0,3})[ ]{0,3}[-到至]{0,3}[ ]{0,3}(\d{4}[ ]{0,3}"
                        r"-[ ]{0,5}\d{1,2}[ ]{0,3}-?[ ]{0,3}\d{0,2}[ ]{0,3})?", text)
        if res is not None:
            start = res.group(1)
            end = res.group(2)
            if start is not None:
                dic['start_time'].append(start)
            else:
                dic['start_time'].append('null')
            if end is not None:
                dic['end_time'].append(end)
            else:
                dic['end_time'].append('null')
        res = re.search(r'(学[ ]{0,10}位|学[ ]{0,10}历|学[ ]{0,10}历\\学[ ]{0,10}位)?'
                        r'[:：\-]?(小[ ]{0,10}学|初[ ]{0,10}中|高[ ]{0,10}中|专[ ]{0,10}科|本[ ]{0,10}'
                        r'科|学[ ]{0,10}士|研[ ]{0,10}究[ ]{0,10}生|硕[ ]{0,10}士|博[ ]{0,10}士)', text)
        if res is not None:
            # print('学历:', res.group(2))
            dic['scholar'].append(res.group(2))


def test0():
    dic = {'name': [], 'gender': [], 'age': [], 'work_time': [], 'place': [], 'city': [], 'birth': [], 'politic': [],
           'contact': [], 'link': [], 'job': [], 'scholar': [], 'school': []}
    texts = ['联系方式：18170702941', 'Email:woeir@qq.com', '学  位\学 历:小学生', '政 治 面 貌——群众', '性 别：男', '微信xqq1226922778)',
             '毕业于杭州电子科技大学', '个人链接：https://www.xuqiqiang.com', '许启强', '籍贯\\户口：中国台湾', '工作单位：高雄市', '23岁']
    texts1 = [
        '应聘职位：', '浙江大学 本科', '硕士：清华大学', '周伟光', 'ID:653228941', '目前正在找工作', '17306810051', 'zhouweiguang@hotmail.com'
    ]
    texts2 = [
        '个人简历',
            '细心从每一个小细节开始。',
            'Personal resume,',
            '个人简历',
            '细心从每一个小细节开始。',
            'Personal resume',
            '周星驰',
            '民族：汉',
            '电话：18817636118',
            '邮箱：1792767532 @ qq.com',
            '住址：杭州市滨江区',
            '性别：男',
            '出生年月：1993.03.01',
            '身高：168cm',
            '学历：硕士研究生',
            '院校：上海大学（211）'
            ]
    extract_item_base_info(texts2, nlp, dic)
    extract_item_by_rule_base_info(texts2, dic)
    print(dic)


def test1():
    # # corpus_processing(ROOT_PATH + '\\base_info.txt')
    # # text = ':hello'
    # # print(seg(text, [':']))
    # file_path = ROOT_PATH + '\\job_order.txt'
    # job_info = load_job(file_path)
    # job_list = []
    # for i in range(job_info.shape[0]):
    #     jobs = job_info.iloc[i, 1]
    #     for job in jobs:
    #         job = re.sub(DEL_SPECIAL_PAT2, '', job)
    #         job_list.append(job)
    # sentences = []
    # labels = []
    # create_job_sentence(job_list, sentences=sentences, labels=labels)
    # # print(sentences[0])
    word2vec = gensim.models.Word2Vec.load(word2vec_model_path_2021_2_5)
    # word2feature(word2vec=word2vec, sentences=sentences, labels=labels)
    # # for i in range(5):
    # #     print(sentences[i])
    # model, crf = create_model()
    # model.compile(optimizer='adam',
    #               loss={'crf_layer': crf.get_loss},
    #               metrics=['accuracy'])
    # # for i, sentence in enumerate(sentences):
    # #     print(sentence)
    # #     print(labels[i])
    # #     print('---------------------------------------------------')
    # # print(sentences[0])
    # sentences = tf.keras.preprocessing.sequence.pad_sequences(sentences, maxlen=8, padding='post', dtype='float', truncating='post')
    # # print('after padding:')
    # # for i in range(5):
    # #     print(sentences[i])
    # # for sen in sentences:
    # #     print(sen)
    # train_x = np.array(sentences)
    # train_x = tf.keras.layers.Masking()(train_x)
    # # print(train_x._keras_mask)
    # labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=8, padding='post',truncating='post')
    # train_y = np.array(labels)
    # # train_y  = tf.keras.layers.Masking()(train_y)
    # print(train_x.shape)
    # print(train_y.shape)
    # model.fit(train_x[:1500], np.argmax(train_y[:1500], axis=-1), batch_size=32, epochs=10)
    # scores = model.evaluate(train_x[1500:], np.argmax(train_y[1500:], axis=-1), verbose=1)
    # print(scores)
    test_text = ['求职意向——数学教师', '担任过销售主管这个职位','期待岗位为数据分析师','金融统计与风险管理方向（研究生）',
                 '浙江大学计算机系','应用统计学专业（本科）','专业方向： 机器学习（ 聚类算法 和 聚类集成）']
    tokenizes = get_tokenize(test_text)
    x = txt2feature_predict(word2vec, tokenizes)
    test_text = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=8, padding='post', dtype='float', truncating='post')
    test_text = np.array(test_text)
    # model.save_weights(JOB_NAME_EXTRACTION_MODEL_PATH)
    model1, _ = create_model()
    model1.load_weights(JOB_NAME_EXTRACTION_MODEL_PATH)
    pre_y = model1.predict(test_text)
    print(pre_y)
    res = get_entity(pre_y, tokenizes)
    print(res)
    # print(scores)

    # freq_analyze(job_info)


def test2():
    file_path = ROOT_PATH + '\\job_order.txt'
    job_info = load_job(file_path)
    stop_words = stopwordslist()
    stop_words.append(' ')
    jieba.load_userdict(USER_DIC_PATH)
    for i in range(job_info.shape[0]):
        text = ' '.join(job_info.iloc[i, 1])
        text = re.sub(DEL_SPECIAL_PAT2, '', text)
        text_list = jieba.lcut(text)
        text_list = [text for text in text_list if text != ' ']
        job_info.iloc[i, 1] = text_list
    with open(ROOT_PATH + '\\job_corpus.txt', 'w', encoding='utf-8') as f:
        res = []
        for i in range(job_info.shape[0]):
            word_list = job_info.iloc[i, 1]
            res.append(' '.join(word_list))
        for r in res:
            print(r)
        f.writelines(res)


def test_extract_edu():
    texts = [
        '2013.06.21 - 2017.5',
        '康奈尔大学',
        '硕士',
        '方向：数据分析与机器学习'
        '2011 / 9 - 2013 / 6',
        '浙江大学',
        '本科',
        '专业：计算机科学与技术'
    ]
    dic = {'school': [], 'start_time': [], 'end_time': [], 'major': [], 'scholar': []}
    model, _ = create_model()
    model.load_weights(JOB_NAME_EXTRACTION_MODEL_PATH)
    word2vec = gensim.models.Word2Vec.load(word2vec_model_path_2021_2_5)
    extract_item_edu(texts=texts, nlp=nlp, dic=dic, word2vec=word2vec, job_model=model)
    extract_item_edu_by_rule(texts, dic)
    print(dic)


if __name__ == '__main__':
    test_extract_edu()
    # test1()







