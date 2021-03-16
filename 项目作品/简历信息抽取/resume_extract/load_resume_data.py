# -*- coding: utf-8 -*-
import os
import win32com.client as wc
from zipfile import ZipFile
import re
import jieba
import format_to_txt
from xml.dom.minidom import parse
import xml.dom.minidom
import pandas
import random
import pandas as pd
from create_data import stopwordslist
from feature_engineer import trans_to_wordvec_by_word2vec

# ROOT_PATH =os.path.abspath(os.path.dirname(__file__))  # 当前所在目录的路径
from special_string import *


def load_data():
    """
    加载多分类模型的数据集
    return:train_data,test_data
    """
    print('feature_engineer:load_data')
    train_file_txt = open(MUTI_CLASSFICATION_TRAIN_DATA_PATH + '\\train_data.txt', 'r', encoding='utf-8').read()
    test_file_txt = open(MUTI_CLASSFICATION_TEST_DATA_PATH + '\\test_data.txt', 'r', encoding='utf-8').read()
    # print(train_file_txt)
    train_list, test_list = train_file_txt.split('###'), test_file_txt.split('###')
    train_frame = pd.DataFrame(columns=['Id','Text','Label'])  # Id,Text,Label
    test_frame = pd.DataFrame(columns=['Id','Text','Label'])  # Id,Text,Label
    id=0
    # print(len(train_list))
    for module in train_list:
        # print(module)
        dic = {"Id":id,"Text": module[1:],"Label":int(module[0])}
        train_frame = train_frame.append(dic, ignore_index=True)
        id += 1
    for module in test_list:
        dic = {"Id": id, "Text": module[1:], "Label": int(module[0])}
        test_frame = train_frame.append(dic, ignore_index=True)
        id += 1
    return train_frame, test_frame


def load_data_for_rnn(data_dir=RNN_DATA_DATA_PATH1, label_dir=RNN_DATA_LABEL_PATH1):
    """

    :param data_dir: 语料所在文件夹路径
    :param label_dir: 简历标签所在文件夹路径
    :return:
    """
    resume_list = []
    label_list = []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    for resume_file_name in resume_file_names:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
        resume_list.append(one_resume)
    for label_file_name in label_file_names:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(i) for i in one_label]
                one_resume_labels.append(one_label)
        label_list.append(one_resume_labels)
    return resume_list, label_list


def load_data_for_rnn_new_add_noise_step(data_dir, label_dir, resume_file_names, label_file_names, noise_percent=0, noise_type='shuffle'):
    x, y = [], []
    indexes = {}  # 用于决定是否产生异常
    for i, resume_file_name in enumerate(resume_file_names):
        is_abnormal = random.randint(0, 9)
        if is_abnormal >= 10 - noise_percent:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
                if len(one_resume) > 2:
                    new_one_resume = []
                    if noise_type == 'skip':  # random skip
                        num = random.randint(0, len(one_resume) - 1)
                        skips = random.sample([k for k in range(len(one_resume))], num)
                        skips = set(skips)
                        for j in range(len(one_resume)):
                            if j in skips:
                                # print('continue')
                                continue
                            new_one_resume.append(one_resume[j])
                        indexes[i] = skips
                    elif noise_type == 'shuffle':  # random shuffle
                        shuffles = random.sample([k for k in range(0, len(one_resume))], len(one_resume))  # 从1开始打乱
                        # change_index = random.randint(1, len(shuffles)-1)
                        # temp = shuffles[change_index]
                        # shuffles[change_index] = 0
                        # shuffles.insert(0, temp)
                        # """
                        # 强制将base info 放到其他位置
                        # one_resume index is [0,1,2,3,4,5]共6个块
                        # eg:shuffles = [2,4,3,5,1]  第一个index 0暂时不打乱
                        # change_index = 1
                        # temp = shuffles[1]=4
                        # shuffles[1] = 0 --> shuffles = [2,0,3,5,1]
                        # shuffles.insert(0, 4) --> shuffles = [4, 2, 0, 3, 5, 1]
                        # """
                        for index in shuffles:
                            new_one_resume.append(one_resume[index])
                        indexes[i] = shuffles
                    else:  # random swap
                        num = random.randint(2, 4) if len(one_resume) >= 4 else random.randint(2, len(one_resume))
                        swaps = random.sample([k for k in range(len(one_resume))], num)  # 随机选取的原始需要打乱的索引子集
                        print(swaps)
                        swaps_dic = {}
                        for j in range(len(swaps)):
                            swaps_dic[swaps[j]] = j

                        shuffles = random.sample(swaps, len(swaps))  # 打乱后的索引
                        for j in range(len(one_resume)):
                            if j in swaps:
                                index_swap = swaps_dic[j]
                                new_one_resume.append(one_resume[shuffles[index_swap]])  # 如果当前索引是需要打乱的，则用打乱后的值填充
                            else:
                                new_one_resume.append(one_resume[j])
                        indexes[i] = (swaps, swaps_dic, shuffles)
                    one_resume = new_one_resume
        else:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
        x.append(one_resume)
    for i, label_file_name in enumerate(label_file_names):
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(k) for k in one_label]
                one_resume_labels.append(one_label)
            if i in indexes.keys():
                skips = indexes[i]
                new_labels = []
                if noise_type == 'shuffle':  # 表明是顺序需要打乱
                    for index in skips:
                        new_labels.append(one_resume_labels[index])
                elif noise_type == 'skip':
                    for j in range(len(one_resume_labels)):
                        if j in skips:
                            continue
                        new_labels.append(one_resume_labels[j])
                else:
                    for j in range(len(one_resume_labels)):
                        if j in indexes[i][0]:
                            index_swap = indexes[i][1][j]
                            new_labels.append(one_resume_labels[indexes[i][2][index_swap]])  # 如果当前索引是需要打乱的，则用打乱后的值填充
                        else:
                            new_labels.append(one_resume_labels[j])
                one_resume_labels = new_labels
        y.append(one_resume_labels)

    return x, y


def load_data_for_rnn_new_add_noise(data_set=1, train_num=25, test_num=400, val_num=None, load_train=True,
                                    train_noise_percent=0, noise_percent=0, noise_type='shuffle'):
    if data_set == 1:
        data_dir = RNN_DATA_DATA_PATH1
        label_dir = RNN_DATA_LABEL_PATH1
    elif data_set == 2:
        data_dir = RNN_DATA_DATA_PATH2
        label_dir = RNN_DATA_LABEL_PATH2
    else:
        data_dir = RNN_DATA_DATA_PATH3
        label_dir = RNN_DATA_LABEL_PATH3
    train_x, train_y, test_x, test_y = [], [], [], []
    val_x, val_y = [], []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    length = len(resume_file_names)
    if load_train:
        train_x, train_y = load_data_for_rnn_new_add_noise_step(data_dir, label_dir, resume_file_names[:train_num],
                                                                label_file_names[:train_num],
                                                                noise_percent=train_noise_percent, noise_type=noise_type)
    test_x, test_y = load_data_for_rnn_new_add_noise_step(data_dir, label_dir, resume_file_names[length - test_num:length],
                                                                label_file_names[length - test_num:length],
                                                                noise_percent=noise_percent, noise_type=noise_type)
    if val_num is not None:
        for resume_file_name in resume_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
            val_x.append(one_resume)
        for label_file_name in label_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
                one_resume_labels = []  # 一份简历中N个标题对应的label
                for one_title_label in f.readlines():
                    # print(one_title_label)
                    one_label = one_title_label.split(' ')
                    one_label = [int(i) for i in one_label]
                    one_resume_labels.append(one_label)
            val_y.append(one_resume_labels)
    if val_num is None:
        if load_train:
            return train_x, train_y, test_x, test_y
        else:
            return test_x, test_y
    else:
        if load_train:
            return train_x, train_y, test_x, test_y, val_x, val_y
        else:
            return test_x, test_y, val_x, val_y


def load_data_for_rnn_new(data_set=1, train_num=25, test_num=400, val_num=None):
    if data_set == 1:
        data_dir = RNN_DATA_DATA_PATH1
        label_dir = RNN_DATA_LABEL_PATH1
    elif data_set == 2:
        data_dir = RNN_DATA_DATA_PATH2
        label_dir = RNN_DATA_LABEL_PATH2
    else:
        data_dir = RNN_DATA_DATA_PATH3
        label_dir = RNN_DATA_LABEL_PATH3
    train_x, train_y, test_x, test_y = [], [], [], []
    val_x, val_y = [], []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    length = len(resume_file_names)
    for resume_file_name in resume_file_names[:train_num]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
        train_x.append(one_resume)
    for label_file_name in label_file_names[:train_num]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(i) for i in one_label]
                one_resume_labels.append(one_label)
        train_y.append(one_resume_labels)

    for resume_file_name in resume_file_names[length - test_num:length]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
        test_x.append(one_resume)
    for label_file_name in label_file_names[length - test_num:length]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            one_resume_labels = []  # 一份简历中N个标题对应的label
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_label = one_title_label.split(' ')
                one_label = [int(i) for i in one_label]
                one_resume_labels.append(one_label)
        test_y.append(one_resume_labels)

    if val_num is not None:
        for resume_file_name in resume_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
            val_x.append(one_resume)
        for label_file_name in label_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
                one_resume_labels = []  # 一份简历中N个标题对应的label
                for one_title_label in f.readlines():
                    # print(one_title_label)
                    one_label = one_title_label.split(' ')
                    one_label = [int(i) for i in one_label]
                    one_resume_labels.append(one_label)
            val_y.append(one_resume_labels)
    if val_num is None:
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y, test_x, test_y, val_x, val_y


def load_data_for_single_muti_classification(data_set=1, train_num=50, test_num=150, val_num=None):
    """
    加载仅对一个模块进行分类的模型所需数据集格式
    :param data_dir: 语料所在文件夹路径
    :param label_dir: 简历标签所在文件夹路径
    :param train_num:用于作为训练集的简历样本数量
    :param test_num:用于作为测试集的简历样本数量
    :return:
    """
    if data_set == 1:
        data_dir = RNN_DATA_DATA_PATH1
        label_dir = RNN_DATA_LABEL_PATH1
    elif data_set == 2:
        data_dir = RNN_DATA_DATA_PATH2
        label_dir = RNN_DATA_LABEL_PATH2
    else:
        data_dir = RNN_DATA_DATA_PATH3
        label_dir = RNN_DATA_LABEL_PATH3
    train_x, train_y, test_x, test_y = [], [], [], []
    val_x, val_y = [], []
    resume_file_names = os.listdir(RNN_DATA_DATA_PATH1)
    label_file_names = os.listdir(RNN_DATA_LABEL_PATH1)
    length = len(label_file_names)
    for resume_file_name in resume_file_names[:train_num]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
            for one_module in one_resume:
                train_x.append(one_module)
    for label_file_name in label_file_names[:train_num]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_title_label = re.sub('\n', '', one_title_label)
                one_label = one_title_label.split(' ')
                no_one_flag = True
                for i, num in enumerate(one_label):
                    if num == '1':
                        train_y.append(i)
                        no_one_flag = False
                        break
                if no_one_flag:
                    print(one_label)
    if val_num is not None:
        for resume_file_name in resume_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
                resume_txt = f.read()
                one_resume = resume_txt.split('###')
                for one_module in one_resume:
                    val_x.append(one_module)
        for label_file_name in label_file_names[length - test_num - val_num:length - test_num]:
            with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
                for one_title_label in f.readlines():
                    # print(one_title_label)
                    one_title_label = re.sub('\n', '', one_title_label)
                    one_label = one_title_label.split(' ')
                    no_one_flag = True
                    for i, num in enumerate(one_label):
                        if num == '1':
                            val_y.append(i)
                            no_one_flag = False
                            break
                    if no_one_flag:
                        print(one_label)
    for resume_file_name in resume_file_names[length - test_num:length]:
        with open(os.path.join(data_dir, resume_file_name), 'r', encoding='utf-8') as f:
            resume_txt = f.read()
            one_resume = resume_txt.split('###')
            for one_module in one_resume:
                test_x.append(one_module)
    for label_file_name in label_file_names[length - test_num:length]:
        with open(os.path.join(label_dir, label_file_name), 'r', encoding='utf-8') as f:
            for one_title_label in f.readlines():
                # print(one_title_label)
                one_title_label = re.sub(r'\n', '', one_title_label)
                one_label = one_title_label.split(' ')
                for i, num in enumerate(one_label):
                    if num == '1':
                        test_y.append(i)
                        break
    if val_num is None:
        return train_x, train_y, test_x, test_y
    else:
        return train_x, train_y, test_x, test_y, val_x, val_y


def load_data_for_word2vec(data_dir=DOWNLOAD_RESUME_PATH1):
    """
    加载训练word2vec所需要的文本语料
    :param data_dir:简历所在文件夹路径
    :return:以列表形式返回,[setence1, sentence2,...]
    """
    setences = []
    file_name_list = os.listdir(data_dir)
    jieba.load_userdict(USER_DIC_PATH)
    stop = stopwordslist()
    length = len(file_name_list)
    temp = ''
    for i, name in enumerate(file_name_list):
        if (i+1) % 500 == 0:
            print(i+1, '/', length)
        path = os.path.join(data_dir, name)
        txt_list = get_docx_txt_with_formal1(path)
        for x in txt_list:
            cut_res = jieba.lcut(re.sub(DEL_SPECIAL_PAT, '', x))
            cut_res = ' '.join([word for word in cut_res if word not in stop])
            temp += ' ' + cut_res
            if len(temp) > 60:
                setences.append(temp)
                temp = ''
    return setences


def one_doc_to_docx(file_path):
    """
    将doc文件转为docx文件
    :param file_path:doc文件所在路径
    :return:在doc文件路径下生成docx文件，无返回
    """
    last_prex = file_path.split('.')[-1]
    if last_prex == 'doc':
        word = wc.Dispatch("Word.Application")
        doc = word.Documents.Open(file_path)
        # 上面的地方只能使用完整绝对地址，相对地址找不到文件，且，只能用“\\”，不能用“/”，哪怕加了 r 也不行，涉及到将反斜杠看成转义字符。
        doc.SaveAs(file_path + r'x', 12, False, "", True, "", False, False, False,
                    False)  # 转换后的文件,12代表转换后为docx文件
        doc.Close()
        word.Quit()
        os.remove(file_path)
        print("remove doc finish,has change to :")
        print(file_path + r'x')


def doc_to_docx(directory_path=None, file_name_list=None):
    if not file_name_list:
        if directory_path is not None:
            file_name_list = os.listdir(directory_path)
            file_name_list = [(directory_path + '\\' + name) for name in file_name_list]
        else:
            print('ERROR docTodocx')
            return
    length = len(file_name_list)
    for i, path in enumerate(file_name_list):
        if i % 3 == 0:
            print('-------------------', i+1, '/', length, '-------------------------')
        last_prex = path.split('.')[-1]
        if last_prex == 'doc':
            # print('original path is doc:', path)
            # pythoncom.CoInitialize()
            try:
                # pythoncom.CoInitial()
                word = wc.Dispatch("Word.Application")
                doc = word.Documents.Open(path)
            # 上面的地方只能使用完整绝对地址，相对地址找不到文件，且，只能用“\\”，不能用“/”，哪怕加了 r 也不行，涉及到将反斜杠看成转义字符。
                doc.SaveAs(path + r'x', 12, False, "", True, "", False, False, False,
                       False)  # 转换后的文件,12代表转换后为docx文件
                doc.Close()
                word.Quit()
                os.remove(path)
                # print("remove doc finish,has change to :")
                # print(path + r'x')
            except:
                print('unknow error happened, pass this file')
                # print(path)
                os.remove(path)
            # pythoncom.CoUninitialize()
    print('finish')


def get_docx_txt(path):
    """
    :param path:
    :return: 返回简历文本内容
    """
    doc = ZipFile(path)
    xml = doc.read('word/document.xml').decode('utf-8')
    text_list = re.findall('<w:t .*?>.*?</w:t>|<w:t>.*?</w:t>', xml)
    txt = ''
    for text in text_list:
        txt += text
    txt = re.sub('<w:t .*?>|<w:t>|</w:t>', '', txt)
    return txt


def get_docx_txt_with_formal(path):
    """
    :param path: docx文档所在路径
    :return: 返回简历文本内容，段落间按换行符隔开
    """
    doc = ZipFile(path)
    xml = doc.read('word/document.xml').decode('utf-8')
    parts = re.findall('<w:p.*?>(.*?)</w:p>', xml)
    # print(parts)
    total_txt = ''
    for part in parts:
        text_list = re.findall('<w:t .*?>.*?</w:t>|<w:t>.*?</w:t>', part)
        txt = ''
        for text in text_list:
            txt += text
        if txt:
            txt = re.sub('<w:t .*?>|<w:t>|</w:t>', '', txt)
            total_txt += '\n' + txt[:]
    return total_txt


def get_docx_txt_with_formal1(path):
    """
    :param path:
    :return:返回简历文本，返回形式为列表，[sentence1, sentence2,...]
    """
    a = path.split('.')[-1]
    if a != 'docx':
        print('load_resume_data:get_docx_txt_with_formal1:not docx,is ', a)
        return []
    doc = ZipFile(path)
    xml = doc.read('word/document.xml').decode('utf-8')
    parts = re.findall('<w:p>(.*?)</w:p>', xml)
    total_txt = []
    for part in parts:
        text_list = re.findall('<w:t .*?>.*?</w:t>|<w:t>.*?</w:t>', part)
        txt = ''
        for text in text_list:
            txt += text
        txt = re.sub('<w:t .*?>|<w:t>|</w:t>', '', txt)
        try:
            total_txt.append(txt[:])
        except IndexError:
            total_txt.append(txt)
    return total_txt


def get_docx_txt_with_formal_table(path):
    # dom解析xml
    doc = ZipFile(path)
    xml1 = doc.read('word/document.xml').decode('utf-8')
    # print(xml1)
    DOMTree = xml.dom.minidom.parseString(xml1)
    # 返回文档的根节点 W:document
    root1 = DOMTree.documentElement
    ContentNodes = root1.getElementsByTagName("w:tbl")
    print('table count：',len(ContentNodes))
    if len(ContentNodes) == 0:
        contentNodes = root1.getElementsByTagName('w:p')
        contents = []
        for node in contentNodes:
            t_nodes = node.getElementsByTagName('w:t')
            txt = ''
            for t_node in t_nodes:
                txt += t_node.firstChild.data
            print(txt)
            contents.append(txt)
        return [contents]
    contents = []
    for i in range(len(ContentNodes)):
        content = []
        tr_nodes = ContentNodes[i].getElementsByTagName('w:tr')
        for tr_node in tr_nodes:
            tc_nodes = tr_node.getElementsByTagName('w:tc')
            tr_node_txt = ''
            for tc_node in tc_nodes:
                p_nodes = tc_node.getElementsByTagName('w:t')
                txt = ''  # 一个tc_node包含的文本
                for p_node in p_nodes:
                    txt += str(p_node.firstChild.data)
                tr_node_txt += ' ' + txt
            content.append(tr_node_txt)
            print(tr_node_txt)
        contents.append(content)

    return contents


def minEditDist(sm, sn):
    m, n = len(sm) + 1, len(sn) + 1

    # create a matrix (m*n)

    matrix = [[0] * n for i in range(m)]

    matrix[0][0] = 0
    for i in range(1, m):
        matrix[i][0] = matrix[i - 1][0] + 1

    for j in range(1, n):
        matrix[0][j] = matrix[0][j - 1] + 1

    "********************"

    cost = 0

    for i in range(1, m):
        for j in range(1, n):
            if sm[i - 1] == sn[j - 1]:
                cost = 0
            else:
                cost = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

    return matrix[m - 1][n - 1]


def get_public_sub_str_len(s1, s2):
    length = 0
    txt = ''
    len1, len2 = len(s1), len(s2)
    for i, x in enumerate(s1):
        temp_len = 0
        for j, y in enumerate(s2):
            if y == x:
                k1, k2 = i, j
                while k1 < len1 and k2 < len2 and s1[k1] == s2[k2]:
                    k1 += 1
                    k2 += 1
                    temp_len += 1
                if temp_len > length:
                    length = temp_len
    return length


def cos_distance(s1, s2):
    """
    计算两字符串之间的余弦距离，不考虑顺序
    :param s1:
    :param s2:
    :return:
    """
    s1 = re.sub(r'和|与|以及|及|求职简历|个人简历|技术部|工程部|[（([【{《<].*?[)）】}》>]|[^0-9\u4E00-\u9FD5]+', '', s1)
    s2 = re.sub(r'和|与|以及|及|求职简历|个人简历|技术部|工程部|[（([【{《<].*?[)）】}》>]|[^0-9\u4E00-\u9FD5]+', '', s2)
    s1 = re.sub(r'简历|部|部门|总监|经理|主管|员|健康|[（([【{《<].*?[)）】}》>]|[^\u4E00-\u9FD5a-zA-Z]+', 'xxxx', s1)
    s2 = re.sub(r'简历|部|部门|总监|经理|主管|员|健康|[（([【{《<].*?[)）】}》>]|[^\u4E00-\u9FD5a-zA-Z]+', 'xxxx', s2)
    dic = {}  # 存放两个字符串中出现的所有字符以及对应的索引号
    count = 0

    for item in s1:
        if item not in dic:
            dic[item] = count
            count += 1
    for item in s2:
        if item not in dic:
            dic[item] = count
            count += 1
    # print('dic is:', dic)
    len1, len2 = len(s1), len(s2)
    vec1, vec2 = [0 for i in range(len(dic))], [0 for i in range(len(dic))]
    for item in s1:
        vec1[dic[item]] += 1
    for item in s2:
        vec2[dic[item]] += 1
    # print("vec1:", vec1)
    # print("vec2:", vec2)
    # 开始计算余弦距离
    AB = 0
    len_v = len(vec1)
    for i in range(len_v):
        AB += vec1[i] * vec2[i]
    # print("AB:", AB)
    ab_val = (sum([x1**2 for x1 in vec1]))**0.5 * (sum([x2**2 for x2 in vec2]))**0.5
    # print("ab_val:", ab_val)
    if len(s1) > 16:
        len_s1 = 16
    elif 4 <= len(s1) <= 14:
        len_s1 = 4 + (len(s1) - 4)/3
    else:
        len_s1 = len(s1)
    if len(s1) <= 1:
        return 0
    try:
        res = AB / round(ab_val, 6) * (1 - abs(len_s1 - 4)/12)  # 长度越靠近4越好
    except ZeroDivisionError:
        return 0
    if 3 <= len(s1) <= 8:
        public_sub_str_len = get_public_sub_str_len(s1, s2)
        similar2 = public_sub_str_len / max(len(s1), len(s2))
        res = res + (1-res) * similar2 * 0.6
    # print("余弦值为:", res)
    # print(s1, ':', s2, '      ', res)
    return res


def get_docx_txt_with_format_list(path, title_list):
    doc = ZipFile(path)
    xml = doc.read('word/document.xml').decode('utf-8')
    parts = re.findall('<w:p .*?>.*?</w:p>|<w:p>.*?</w:t>', xml)
    total_txt = []
    for part in parts:
        text_list = re.findall('<w:t .*?>.*?</w:t>|<w:t>.*?</w:t>', part)
        if len(text_list) == 0:
            continue
        # print("text_list:", text_list)
        txt = ''
        for text in text_list:
            txt += text
        txt = re.sub('<w:t .*?>|<w:t>|</w:t>', '', txt)
        # 本身属性
        key_similarity = 0
        key_title = ''
        for title in title_list:
            sim = cos_distance(txt, title)
            if key_similarity < sim:
                key_similarity = sim
                key_title = title
        if key_similarity > 0.3:
            title = ' '.join(title_list[key_title])
        else:
            title = 'NAN'
        w_r_num = len(text_list)  # 一个段落包含的文本数量
        txt1 = re.sub(r'[^\u4E00-\u9FD5a-zA-Z\d]+', '', txt)
        text_len = len(txt1)  # 段落长度
        special_char_num = len(re.findall('[,。;；，、:："]', txt))
        whitespaces = re.findall('[^ \u3000]( +?)[^ \u3000]', txt)
        num = sum([len(hehe) for hehe in whitespaces])
        if key_similarity >= 0.5:
            num //= 3
        brackets_num = find_bracket_num(txt)
        special_char_num += brackets_num + num
        # 格式属性
        bold_it = 0
        bold_res = re.findall('<w:b.*?/>|<w:i.*?/>', part)
        if bold_res and len(bold_res) >= 0.8 * w_r_num:
            bold_it = 1

        font_size_res = re.findall('w:sz(.*?)/>', part)
        # print("font_size_txt_list:", font_size_res)
        if font_size_res:
            font_size = 0
            for item in font_size_res:
                font_size += float(re.search('[\d]+',  item).group())
            font_size /= len(font_size_res)
        else:
            font_size = -1  # 缺省，用平均值代替
        # print('font-size:', font_size)
        color = re.findall(r'<w:color.*?w:val="(.*?)".*?/>', part)
        if len(color)==0:
            color = "NAN"
            # print("color:", color)
        else:
            # print("color:", color[0])
            color = max(color, key=color.count)
        feature = {"Text_len": text_len, "W_r_num": w_r_num,  "Special_char_num": special_char_num,
                   "Bold_italic": bold_it, "Font_size": font_size,
                   "Color": color, 'Title': title, "Text": txt,"Text1":txt1, "Key_similarity": key_similarity}
        total_txt.append(feature)
    length = len(total_txt)
    color_list = [x['Color'] for x in total_txt if x['Color'] != 'NAN']
    color_forbid = False
    if color_list:
        color_k = max(color_list, key=color_list.count)
        count = 0
        for x in color_list:
            if x == color_k:
                count += 1
        if count / len(color_list) >= 0.8:
            color_forbid = True
    else:
        color_forbid = True
    for i, txt_f in enumerate(total_txt):
        # 上一个段落的属性
        if i != 0:
            pre_color = total_txt[i-1]['Color']  #
            previous_font_size = total_txt[i-1]['Font_size']  # font_size
            previous_bold = total_txt[i-1]['Bold_italic']  # bold index
            previous_text_len = total_txt[i-1]['Text_len']
            previous_key_similarity = total_txt[i-1]['Key_similarity']
        else:
            pre_color = txt_f['Color'] + '1'
            previous_font_size = txt_f['Font_size'] * 0.8
            previous_bold = 0
            previous_text_len = random.randint(60, 100)
            previous_key_similarity = random.uniform(0.01, 0.3)
        # 下一个段落的相关属性
        if i != length-1:
            next_color = total_txt[i+1]['Color']
            next_font_size = total_txt[i+1]['Font_size']
            next_bold = total_txt[i+1]['Bold_italic']
            next_text_len = total_txt[i+1]['Text_len']
            next_key_similarity = total_txt[i +1]['Key_similarity']
        else:
            next_color = txt_f['Color']
            next_font_size = txt_f['Font_size']
            next_bold = 1
            next_text_len = random.randint(0, 10)
            next_key_similarity = random.uniform(0.6, 1)
        # 上下文属性
        color_diff = 3  # 0表示上、下文均与本段落相同,1表示上下文有一个与本段落不同，2表示均不同
        bold_diff = 2 # 0表示上下文均与本段落均不同，1表示有一个与本段落不同，2表示均不同
        if txt_f['Color'] == pre_color:
            color_diff -= 1
        if txt_f['Color'] == next_color:
            color_diff -= 1
        if pre_color != next_color:
            color_diff -= 1
        if txt_f['Bold_italic'] == previous_bold:
            bold_diff -= 1
        if txt_f['Bold_italic'] == next_bold:
            bold_diff -= 1
        if txt_f['Bold_italic'] == 0:
            bold_diff = 0
        if 'NAN' in [pre_color, txt_f['Color'], next_color]:
            color_forbid = True
        # 计算差异因子

        similarity_diff = 0.5*(txt_f['Key_similarity'] - (next_key_similarity + previous_key_similarity)*0.5 + 1)
        if similarity_diff >= 0.7:
            similarity_diff = 0.55
        elif similarity_diff >= 0.6:
            similarity_diff = 0.45
        elif similarity_diff > 0.5:
            similarity_diff = 0.25
        elif similarity_diff == 0.5:
            similarity_diff = 0.1
        else:
            similarity_diff = 0
        length_diff = (next_text_len + previous_text_len) * 0.5 - txt_f['Text_len']
        if len(txt1) <= 12:
            if length_diff >= 15:
                length_diff = 0.45
            elif length_diff >= 10:
                length_diff = 0.25
            elif length_diff >= 5:
                length_diff = 0.1
            elif length_diff >= 0:
                length_diff = 0.05
            else:
                length_diff = 0
        else:
            length_diff = 0
        font_size_diff = 0
        if previous_font_size != -1 and txt_f['Font_size'] != -1 and next_font_size != -1:
            font_size_diff = txt_f['Font_size'] - (next_font_size + previous_font_size) * 0.5
            if font_size_diff > 10:
                font_size_diff = 0.45
            elif font_size_diff >= 5:
                font_size_diff = 0.4
            elif font_size_diff > 2:
                font_size_diff = 0.35
            elif font_size_diff > 1:
                font_size_diff = 0.3
            elif font_size_diff > 0:
                font_size_diff = 0.25
            else:
                font_size_diff = 0
        elif (previous_font_size == -1 or next_font_size == -1) and txt_f['Font_size'] != -1:
            font_size_diff = 0.3
        if previous_font_size == -1 and next_font_size == -1 and txt_f['Font_size'] != -1:
            font_size_diff = 0.4
        if 'NAN' not in [pre_color, txt_f['Color'], next_color]:
            if color_diff == 3:
                color_diff = 0.4
            elif color_diff == 2:
                color_diff = 0.3
            elif color_diff == 1:
                color_diff = 0.25
            else:
                color_diff = 0
        elif (pre_color == 'NAN' or next_color == 'NAN') and txt_f['Color'] != 'NAN':
            color_diff = 0.3
        if pre_color == 'NAN' and next_color == 'NAN' and txt_f['Color'] != 'NAN':
            color_diff = 0.4
        if bold_diff == 3:
            bold_diff = 0.4
        elif bold_diff == 2:
            bold_diff = 0.3
        elif bold_diff == 1:
            bold_diff = 0.25
        else:
            bold_diff = 0
        if txt_f['W_r_num'] > 4:
            txt_f['W_r_num'] = 1
        elif txt_f['W_r_num'] != 0:
            txt_f['W_r_num'] = (txt_f['W_r_num'] - 1) / 3
        if txt_f['Special_char_num'] >= 6:
            txt_f['Special_char_num'] = 1
        else:
            txt_f['Special_char_num'] /= 6

        format_f = 0.1 * (1 - txt_f['W_r_num']) + 0.9 * (1 - txt_f['Special_char_num'])
        if not color_forbid:
            difference = min(similarity_diff + bold_diff + font_size_diff + color_diff + length_diff, 1)
        else:
            difference = min(similarity_diff + bold_diff + font_size_diff + length_diff, 1)
        # print('diff item: ', similarity_diff, bold_diff, font_size_diff, length_diff)
        diff_forbid = False
        if txt_f['Key_similarity'] >= 0.6 and (next_key_similarity >= 0.65 or previous_key_similarity >= 0.65):
            diff_forbid = True  # 当相邻的两段文本中有一个可能是标题，则上下文相似度差异不再适用,此时只看文本相似度
        txt_f.update({"Similarity_diff":similarity_diff,"Difference": difference,'Format_f':format_f})
        # 决定标签
        if format_f >= 0.9:
                if diff_forbid:
                    if txt_f['Key_similarity'] >= 0.75:
                        score = 1
                    else:
                        score = 0
                else:
                    if txt_f['Key_similarity'] >= 0.71 and difference >= 0.55:
                        score = 1
                    else:
                        score = 0
        elif format_f >= 0.8:
                if diff_forbid:
                    if txt_f['Key_similarity'] >= 0.8:
                        score = 1
                    else:
                        score = 0
                else:
                    if txt_f['Key_similarity'] >= 0.74 and difference >= 0.55:
                        score = 1
                    else:
                        score = 0
        elif format_f >= 0.7:
                if diff_forbid:
                    if txt_f['Key_similarity'] >= 0.85:
                        score = 1
                    else:
                        score = 0
                else:
                    if txt_f['Key_similarity'] >= 0.77 and difference >= 0.55:
                        score = 1
                    else:
                        score = 0
        else:
            score = 0
            """
            score = txt_f['Key_similarity'] * 0.7 + format_f * 0.15 + difference * 0.15
        else:
            if difference > 0.75:
                score = txt_f['Key_similarity'] * 0.45 + format_f * 0.2 + difference * 0.35
            elif difference > 0.6:
                score = txt_f['Key_similarity'] * 0.5 + format_f * 0.25 + difference * 0.25
            else:
                if format_f > 0.8
                """
        if score > 0.8:
            txt_f['Label'] = 1
        else:
            txt_f['Label'] = 0
        if 0.5 <= txt_f['Key_similarity'] <= 1:
            pass
            # print('title:', txt_f['Text'], ' similarity:', txt_f['Key_similarity'], ' diff:', txt_f['Difference'],
            #       ' special_num:', txt_f['Special_char_num'], ' label:', txt_f['Label'])
            # print(txt_f)
    return pandas.DataFrame(total_txt)


def find_bracket_num(txt):
    bracket_map = {')': '(', ']': '[', '}': '{', '）': '（', '】': '【', '>': '<'}
    brackets = ('(', '[', '{', '（', '【', '<')
    count = 0
    stack = []
    for word in txt:
        if word in brackets:
            stack.append(word)
        elif word in bracket_map:
            if stack and stack[-1] == bracket_map[word]:
                count += 1
                stack.pop()
    return count


def find_locate(text, ch):
    index = []
    for i in range(len(text)):
        if text[i] == ch:
            index.append(i)
    return index


def get_distance(begin, j, text1):
    text = text1[:j]
    lens_1 = len(text1)
    lens = len(text)
    pre = begin
    while begin < lens and j < lens_1 and text[begin] == text1[j]:
        begin += 1
        j += 1
    return begin - pre


def get_between_distance(distance):  # 根据重复文本的长度，得出至少应该间隔多少距离才算非重复文本
    if distance <=1:
        return 0
    elif distance <=3:
        return 3
    elif distance <=12:
        return 5
    else:
        return 50


def text_filter(text, distance=15):
    """ 对文本进行过滤，去除重复内容
    """
    temp_txt = ''
    i = 0
    while i < len(text):
        begin_list = find_locate(text[:i], text[i])  # 如果之前存在，则从这个位置起进行跳过
        if begin_list:  # 表示存在
            max_distance = 0
            max_begin = 0
            for begin in begin_list:  # 计算当前字符开始和之前文本中出现的最大重复长度
                temp_distance = get_distance(begin, i, text)
                if temp_distance > max_distance:
                    max_distance = temp_distance
                    max_begin = begin  # 记录最大重复文本的起始位置
            max_end = max_begin + max_distance - 1
            dis_between = len(text[:i]) - max_end
            if max_distance < distance:  # 当最大重复长度小于规定的distance时认为不属于重复内容，需要加入结果中
                if dis_between > get_between_distance(max_distance):  # 当距离上一次重复文本起始位置的距离足够大时我们才认为这不是重复内容
                    for k in range(i, i + max_distance):
                        temp_txt += text[k]
            i += max_distance - 1
        else:
            temp_txt += text[i]
        i += 1
    return temp_txt


def get_txt_list(path):
    """
    将pdf或docx格式文档提取为文本格式文档
    param@path:原docx或pdf文件数据所在文件夹路径
    return:文本列表
    """
    file_names = os.listdir(path)
    txt_list = []
    pdf2Txter = format_to_txt.CPdf2TxtManager()
    for i in range(len(file_names)):
        last_prex = file_names[i].split('.')[-1]
        if last_prex != 'pdf':
            # print('not pdf:', file_names[i])
            txt = get_docx_txt(path + '\\' + file_names[i])
            txt = re.sub(' ', '', txt)
            if len(txt) <= 10:
                print('content is:')
                print(txt)
        else:
            # print('is pdf:', file_names[i])
            txt = pdf2Txter.convert_pdf_to_txt(path + '\\' + file_names[i])
            txt = re.sub(' ', '', txt)
            if len(txt) <= 10:
                print('content is:')
                print(txt)
        txt_list.append(txt)
    # print('txt_list.length:', len(txt_list))
    return txt_list


def get_txt_list_with_format(path,title_list):
    """
        将pdf或docx格式文档提取为文本格式文档
        param@path:原docx或pdf文件数据所在文件夹路径
        return:文本列表
    """
    file_names = os.listdir(path)
    txt_list = pandas.DataFrame()
    pdf2Txter = format_to_txt.CPdf2TxtManager()
    for i in range(len(file_names)):
        last_prex = file_names[i].split('.')[-1]
        if last_prex != 'pdf':
            # print('not pdf:', file_names[i])
            txt_formats = get_docx_txt_with_format_list(path + '\\' + file_names[i], title_list=title_list)
            txt_list = txt_list.append(txt_formats)
        """else:
            # print('is pdf:', file_names[i])
            txt = pdf2Txter.convert_pdf_to_txt(path + '\\' + file_names[i])
            txt = re.sub(' ', '', txt)
            if len(txt) <= 30:
                print('content is:')
                print(txt)
        txt_list.append(txt)"""
    print('txt_list.length:', txt_list.shape)
    return txt_list


def get_txt_list_from_file():
    train_file_names = os.listdir(TRAIN_TXT_PATH)  # 训练集文本
    test_file_names = os.listdir(TEST_TXT_PATH)  # 测试集文本
    train_list = []
    test_list = []
    for train_name in train_file_names:
        path = TRAIN_TXT_PATH + '\\' + train_name
        f = open(path, 'r', encoding='utf-8')
        txt = f.read()
        train_list.append(txt)
    for test_name in test_file_names:
        path = TEST_TXT_PATH +'\\'+ test_name
        f = open(path, 'r', encoding='utf-8')
        txt = f.read()
        test_list.append(txt)
    return train_list, test_list


def load_title_list(path):
    res = {}
    f = open(path,'r',encoding='utf-8')
    for line in f.readlines():
        # print(line)
        line = re.sub(r'\n|\ufeff','',line)
        line_l = line.split(' ')
        key, key_item = line_l[0], line_l[1:]
        res[key] = key_item
    f.close()
    return res


def load_word2vec_resume_corpus(path=sentence_51job_resume_corpus_path):
    """
    将用于word2vec训练的语料加载为列表形式
    :param path:
    :return: 列表 [sentence1, sentence2, ...]
    """
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # print(lines)
        for line in lines:
            sentences.append(line)
    return sentences


def load_word2vec_wiki_corpus(path=sentence_zhwiki_corpus_txt_path, add_count=10000, last=0):
    """
    每次加载add_count行
    :param path:word2vec语料所在路径
    :param add_count:每次迭代加载的行数
    :param last:上一次迭代结束的位置，即本次迭代的起始位置
    :return:返回sentence列表
    """
    import linecache
    finish_flag = False
    while not finish_flag:
        sentences = []
        for i in range(add_count):
            line = linecache.getline(path, last + i + 1)
            if line:
                sentences.append(line[:-1].split(' '))
            else:
                finish_flag = True
        linecache.clearcache()
        last += add_count
        yield sentences


def load_item_exp_txt(dir):
    file_names = os.listdir(dir)
    for name in file_names:
        if int(name.split('_')[-1][0]) == 9:
            print('###')
            with open(os.path.join(dir, name), 'r', encoding='utf-8') as f:
                print(f.read())


if __name__ == '__main__':
    def test():
        txt = 'eirgoirjgoi{哈哈哈})【这是什么东西haiejgoie（i埃尔价格i）（argj）（）（aeoirgj）（）（ajerogi（）（aerg）（aer））】'
        print(find_bracket_num(txt))


    def test2():
        title_list = load_title_list(TITLE_LIST_PATH)
        resume_txt = get_txt_list_with_format(MATERIAL_TRAIN_PATH, title_list=title_list)

        resume_txt.to_csv(TITLE_REC_DATA_PATH + '\\title_data.csv')
        print(resume_txt)


    def test3():
        title_list = load_title_list(TITLE_LIST_PATH)
        test_str = ['受教育情况145', '教育背景', '__..?技能状况__.*', "个人相关信息：", '留学经验', '基本信息', '疾风剑豪？']
        res = []
        for item in test_str:
            res.append(max([cos_distance(item, x) for x in title_list]))
        print(res)


    def test_doc2docx_all():
        from threading import Thread
        file_name_list = os.listdir(DOWNLOAD_RESUME_PATH1)
        file_name_list = [os.path.join(DOWNLOAD_RESUME_PATH1, name) for name in file_name_list if
                          name.split('.')[1] == 'doc']
        # print(file_name_list)
        print(len(file_name_list))
        thread_num = 32
        slice_num = len(file_name_list) // thread_num
        for i in range(thread_num - 1):
            file_list = file_name_list[i * slice_num:(i + 1) * slice_num]
            t = Thread(target=doc_to_docx, kwargs={'file_name_list': file_list}, name='thread' + str(i + 1))
            t.start()
        file_list = file_name_list[(thread_num - 1) * slice_num:len(file_name_list)]  # 处理余数
        t = Thread(target=doc_to_docx, kwargs={'file_name_list': file_list}, name='thread' + str(thread_num))
        t.start()


    def test_doc2docx_all1():
        file_name_list = os.listdir(DOWNLOAD_RESUME_PATH1)
        # file_name_list = [os.path.join(DOWNLOAD_RESUME_PATH1, name) for name in file_name_list if
        #                   name.split('.')[-1] == 'doc']
        # print(file_name_list)
        print(len(file_name_list))
        # doc_to_docx(file_name_list=file_name_list)
        # doc_to_docx(DOWNLOAD_RESUME_PATH1)


    def test_doc2docx():
        file_path = 'D:\\Download\\简历模板.doc'
        one_doc_to_docx(file_path)


    def test_re():
        txt = '教育背景　                                               　　  '
        txt1 = '教育背景　                                               　　  '
        for item in txt1:
            print(repr(item))
        whitespaces = re.findall('[^ ]( +?)[^ ]', txt1)
        num = sum([len(item) for item in whitespaces])
        print(num)


    def test_get_public_sub_str():
        s1 = ''
        s2 = ''
        print(get_public_sub_str_len(s1, s2))

    def test_load_data_for_word2vec():
        res = load_data_for_word2vec(DOWNLOAD_RESUME_PATH1)
        res1 = res[:500]
        res = []
        for item in res1:
            print(item)

    def test_save_word2vec_data():
        res = load_data_for_word2vec(DOWNLOAD_RESUME_PATH1)
        with open(sentence_51job_resume_corpus_path, 'w', encoding='utf-8') as f:
            for item in res:
                f.write(item)
                f.write('\n')
            f.close()

    def test_get_docx_txt_with_formal1():
        path = DOWNLOAD_RESUME_PATH1 + '\\' + '229.docx'
        txt = get_docx_txt_with_formal1(path)
        for item in txt:
            print(item)

    def test_load_word2vec_resume_corpus():
        sen = load_word2vec_resume_corpus()
        print(len(sen))

    def test_load_word2vec_wiki_corpus():
        generator = load_word2vec_wiki_corpus(add_count=5000)
        for item in generator:
            print('-----------------------------------------------------------------------')
            print(len(item))
            print('')
            print('')

    def test_load_item_exp_txt():
        load_item_exp_txt(MUTI_CLASSFICATION_TRAIN_DATA_PATH)

    def test_save_resume_txt_all():
        txts = get_txt_list(MATERIAL_TRAIN_PATH)
        for i, txt in enumerate(txts):
            with open(TRAIN_TXT_PATH + '\\' + str(i) + '.txt', 'w', encoding='utf-8') as f:
                f.write(txt)

    def test_load_data():
        train_frame, test_frame = load_data()
        print(train_frame)


    def test_5():
        str1 = """0
            姓名:干马群 性别：男 政治面貌： 电话：18871433770 邮箱：fRyMFc@yahoo.cn 社交信息：微博:smSfjxC QQ:1899288175 博客:http://juejin.im/vBYt5lpQD8Rbwa/ 知乎:dGfAB9PUVgRTJ2mM
            ###5
             国家级奖学金 大学三等奖学金 软件杯二等奖 互联网+竞赛全国三等奖 IT培训结业证书 中级软件开发工程师 Kaggle天池赛第二名 ACM世界银奖
            ###0
            姓名:左被山 性别：男 所属民族:回族 毕业学校：首都师范大学   电话：+81 17539189535 邮箱：0UoHOZG8@nuaa.com
            ###0
            姓名:曾勇 政治面貌：  联系电话：+86 13998674148
            ###0
            姓名:方启 籍贯:甘肃 毕业学校：浙江师范大学 群众  邮箱：oS9KxWG7cZDpj@123.com
            ###6
            在整个过程中，并没有物质的转移，只有手掌之间的相对运动"""
        str2 = 'aoijgoier###jgisaerjgioje###ghoiaghiojerio###jiowaergjoi;;;jaioerjhoi###jrjthoijrthio'
        arr_list = '###'.split(repr(str2))
        print(arr_list)

    def test_load_data_for_rnn():
        resume_list, label_list = load_data_for_rnn()
        resume_list = trans_to_wordvec_by_word2vec(resume_list, feature_size=100, word2vec_model=word2vec_model_path_zhwiki_update_814, type='rnn')
        print(resume_list)


    def test_load_data_for_single_muti_classification():
        train_x, train_y, test_x, test_y = load_data_for_single_muti_classification(data_set=1, train_num=50, test_num=150)
        train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=100,
                                                   word2vec_model=word2vec_model_path_zhwiki_update_814, type='full')
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                                               word2vec_model=word2vec_model_path_zhwiki_update_814, type='full')
        # for txt in resume_list:
        #     print(txt)
        print(len(train_x), len(test_x))


    # ----------------------------------------测试-------------------------------------- #
    # test_doc2docx()
    # test_doc2docx_all1()
    # test_re()
    # test_get_public_sub_str()
    # test_load_data_for_word2vec()
    # test_save_word2vec_data()
    # test_get_docx_txt_with_formal1()
    # test_load_word2vec_resume_corpus()
    # test_load_word2vec_wiki_corpus()
    # test_load_item_exp_txt()
    # test_save_resume_txt_all()
    # test_load_data()
    # test_5()
    # test_load_data_for_rnn()
    # test_load_data_for_single_muti_classification()
    train_x, train_y, test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=25, test_num=400, val_num=None)
    for one_resume in test_y:
        print('------------------------------')
        for module in one_resume:
            print(module)
