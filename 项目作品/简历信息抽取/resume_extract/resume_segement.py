"""
首先，基于关键字进行切分，得到一个基本的模块划分，默认该方案的正确概率为0.5
其次，采用多分类器对每一种类的模块训练，得出每一模块的所属类别
若两种方案分类一致，则划分正确，若不一致，则选择max(p(方案一）,p(方案2))所对应的方案所判定的类别
"""
import re
import random
from special_string import *
from load_resume_data import get_txt_list_from_file
from load_resume_data import get_docx_txt_with_formal, get_docx_txt_with_format_list,get_txt_list
from create_data import load_school, create_other_data, create_muti_classification_data
import os
import tensorflow.keras as kr
from feature_engineer import trans_to_wordvec_by_word2vec
from static_num import *
import gensim
label_number = {'base_info': 0, 'edu_back': 1, 'job_exp': 2, 'self_comment': 3, 'school_exp': 4, 'honour': 5,
           'other': 6, 'sci_exp': 7,'skill': 8,'item_exp':9}

number2label = ['base_info', 'edu_back', 'job_exp', 'self_comment', 'school_exp', 'honour',
           'other', 'sci_exp','skill','item_exp']


def get_resume_data():
    return get_txt_list_from_file()


def get_resume_data1():
    return get_txt_list(MATERIAL_TRAIN_PATH)


def get_one_resume_txt(path):
    """
    从指定文件(word/pdf)提取文本
    :param path:文件所在路径
    :return:段落列表
    """
    return get_docx_txt_with_formal(path)


def sort_title(title_list):
    """
    """
    i = 0
    title_list = delete_same(title_list=title_list, begin=0, end=len(title_list) - 1)
    length = len(title_list)
    while i < length:
        first_c = title_list[i][0]
        if i+1 < length:
            while title_list[i + 1][0] == first_c:  # 当下标i+1的title首字母不等于first_c时跳出循环
                i += 1
            j = i + 2
            while j < length:  # 从首字和前面不相等的那一个标题的后一个开始一直遍历到最后一个，只要有等于first_c的，则交换，直到没有为止
                if title_list[j][0] == first_c:
                    temp = title_list[i+1]
                    title_list[i+1] = title_list[j]
                    title_list[j] = temp
                    i += 1
                j += 1
        i += 1
    return title_list


def delete_same(title_list, begin, end):
    """
    指定区间 去重复
    """
    if not title_list:
        return []
    res = []
    # print(title_list[begin: end+1])
    if begin == end:
        # print('begin = end:',begin,'=',end)
        res.append(title_list[begin])
        # print('now the res list is :',res)
        return res
    mid = (end + begin)//2
    title_list1 = delete_same(title_list, begin, mid)
    title_list2 = delete_same(title_list, mid + 1, end)
    for i in range(len(title_list1)):
        res.append(title_list1[i])
    for item in title_list2:
        if item not in res:
            res.append(item)
    # print('after merge:',res)
    return res


def segment_one_resume(txt, pat):
    """
    将一个文本文档切割成模块，字典形式，模块间无序
    """
    dic = {}  # 存放找出的模块
    if len(txt) == 0:
        return dic
    segment = re.compile(pat, re.I)
    res = re.finditer(segment,txt)
    loc_list = []
    title_list = []
    for item in res:
        loc_list.append(item.span())
        title_list.append(item.group())
    for i in range(len(loc_list)-1):
        dic[title_list[i]] = txt[int(loc_list[i][1]):int(loc_list[i+1][0])]
    if loc_list:
        dic['基本信息'] = txt[0:int(loc_list[0][0])]
        if loc_list[len(loc_list)-1][1] < len(txt):
            dic[title_list[len(title_list)-1]] = txt[int(loc_list[len(loc_list)-1][1]):]
    return dic


def segment_one_resume1(txt, pat):
    """
    将一个文本文档切割成模块,列表形式，保持模块间有序
    """
    dic = []  # 存放找出的模块
    if len(txt) == 0:
        return dic
    segment = re.compile(pat, re.I)
    res = re.finditer(segment,txt)
    loc_list = []
    title_list = []
    for item in res:
        print(item.span())
        loc_list.append(item.span())
        title_list.append(item.group())
    for i in range(len(loc_list)-1):
        dic.append([title_list[i], txt[int(loc_list[i][1]):int(loc_list[i+1][0])]])
    if loc_list:
        if loc_list[len(loc_list)-1][1] < len(txt):
            dic.append([title_list[len(title_list)-1], txt[int(loc_list[len(loc_list)-1][1]):]])
        dic.insert(0, ['基本信息', txt[0:int(loc_list[0][0])]])
    return dic


def merge_to_new_module(segment_res, title_dic):
    """
    将初步划分的模块整合成按照固定分类的模块
    segment_res:字典，{标签:文本内容,...}
    title_dic:标签对应类别，字典，形如:{'基本信息':['base_info'],...}
    返回：固定标签的模块字典，形如：{'base_info':'...','edu_back':'...',...}，字典关键字固定
    """
    dic = {'base_info': '', 'edu_back': '', 'job_exp': '', 'self_comment': '', 'school_exp': '', 'honour': '',
           'other': '', 'sci_exp': '', 'skill': '', 'item_exp': ''}
    for item in segment_res:
        if item in title_dic:
            for label in title_dic[item]:
                dic[label] += segment_res[item] + '\n'
    return dic


def create_key_feature(txt, title_list):
    pat = '|'.join(title_list)
    res = re.search(pat, txt[0])
    if res:
        txt[0] = 1


def pass_key_new(txt, title_list, key_model):
    """
    经过构造关键词特征后，特征为[文本是否包含在列表中的关键词，文本长度，文本与列表中关键词的相似度， 是否有下划线， 是否加粗， 是否斜体， 字体大小， 包含的关键词附近是否存在特殊字符]
    :param txt: [text,font1,font2,...]
    :param title_list:title key list
    :return:True or False
    """
    create_key_feature(txt, title_list)
    # 模型预测是否为标题关键字
    res = key_model.predict(txt)
    if res == 1:
        return True
    else:
        return False


def load_models(file_path):
    return kr.models.load_model(file_path)


def segment_one_resume_list_format(txt_list, nn_model_w=None, word2vec_model=None):
    """
    
    :param txt_list:DataFrame [ [txt,font1,font2,...,label,title],   [txt,font1,font2,...,label,title],  ...]
    :param title_list :title key list
    :return: [[key,txt], [key,txt],...]
    """""
    length = len(txt_list)
    modules = []
    key_loc = 0
    last_key = 'base_info'
    one_module = ''
    if nn_model_w is None:
        nn_model_w = load_models(muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec)
    # nn_model_w.build((None, 350, 100))
        nn_model_w.summary()
    if word2vec_model is None:
        word2vec_model = gensim.models.word2vec.Word2Vec.load(word2vec_model_path_zhwiki_rnn_update_20_923)
    for i in range(length):
        if txt_list.iloc[i, :]['Label'] == 0:
            one_module += '\n' + txt_list.iloc[i, :]['Text']
        else:
            text_array = trans_to_wordvec_by_word2vec([one_module], feature_size=WORD2VEC_FEATURE_NUM, type='cnn', max_len=MAX_LEN,
                                                      word2vec_model=word2vec_model)  # 转为词向量
            res = nn_model_w.predict(text_array)
            # print(res[0])
            res_list = list(res[0])
            # print(res_list)
            tag = number2label[res_list.index(max(res[0]))]
            # print(tag)
            # print(one_module[1:])
            modules.append([tag, one_module[1:]])
            one_module = ''
            key_loc = i
            last_key = txt_list.iloc[i, :]['Title']
    if key_loc != length-1:
        modules.append([last_key, one_module[1:]])
    return modules


def segment_one_resume_list_format_without_title(txt_list):
    """

    :param txt_list:DataFrame [ [txt,font1,font2,...,label,title],   [txt,font1,font2,...,label,title],  ...]
    :param title_list :title key list
    :return: [[key,txt], [key,txt],...]
    """""
    length = len(txt_list)
    modules = []
    key_loc = 0
    one_module = ''
    for i in range(length):
        if txt_list.iloc[i, :]['Label'] == 0:
            one_module += '\n' + txt_list.iloc[i, :]['Text']
        else:
            modules.append(one_module[1:])
            one_module = ''
            key_loc = i
    if key_loc != length - 1:
        modules.append(one_module[1:])
    return modules


def segment_all_files(train_path, test_path, title_list, nn_model_w=None, word2vec_model=None):
    """
    将所有简历文件分块
    :param path:
    :return:
    """
    if not nn_model_w:
        nn_model_w = load_models(muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec)
    # nn_model_w.build((None, 350, 100))
        nn_model_w.summary()
    if not word2vec_model:
        word2vec_model = gensim.models.word2vec.Word2Vec.load(word2vec_model_path_zhwiki_update_814)
    all_file_modules = []
    file_names = os.listdir(train_path)
    for file_name in file_names:
        last_prex = file_name.split('.')[-1]
        if last_prex == 'docx':
            file_path = train_path + '\\' + file_name
            total_txt = get_docx_txt_with_format_list(file_path, title_list)
            modules = segment_one_resume_list_format(txt_list=total_txt, nn_model_w=nn_model_w, word2vec_model=word2vec_model)
            all_file_modules.append(modules)
    file_names = os.listdir(test_path)
    for file_name in file_names:
        last_prex = file_name.split('.')[-1]
        if last_prex == 'docx':
            file_path = test_path + '\\' + file_name
            total_txt = get_docx_txt_with_format_list(file_path, title_list)
            modules = segment_one_resume_list_format(txt_list=total_txt,nn_model_w=nn_model_w, word2vec_model=word2vec_model)
            all_file_modules.append(modules)
    """for item in all_file_modules[0]:
        print(item[0])
        print(item[1])
        print('**********************************************************************************************')
        """
    return all_file_modules


def segment_all_files1(train_path, title_list, nn_model_w=None, word2vec_model=None):
    """
    将所有简历文件分块
    :param path:
    :return:
    """
    if not nn_model_w:
        nn_model_w = load_models(muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec)
    # nn_model_w.build((None, 350, 100))
        nn_model_w.summary()
    if not word2vec_model:
        word2vec_model = gensim.models.word2vec.Word2Vec.load(word2vec_model_path_zhwiki_update_814)
    all_file_modules = []
    file_names = os.listdir(train_path)
    for file_name in file_names:
        last_prex = file_name.split('.')[-1]
        if last_prex == 'docx':
            file_path = train_path + '\\' + file_name
            total_txt = get_docx_txt_with_format_list(file_path, title_list)
            modules = segment_one_resume_list_format(txt_list=total_txt,nn_model_w=nn_model_w, word2vec_model=word2vec_model)
            # print("******************************************************************************************")
            all_file_modules.append(modules)
    # for item in all_file_modules[0]:
    #     print(item[0])
    #     print(item[1])
    #     print('**********************************************************************************************')
    # return all_file_modules


def segment_all_files2(file_paths, title_list,nn_model_w=None, word2vec_model=None):
    """
    将所有简历文件分块
    :param path:
    :return:
    """
    if nn_model_w is None:
        nn_model_w = load_models(muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec)
    # nn_model_w.build((None, 350, 100))
        nn_model_w.summary()
    if word2vec_model is None:
        word2vec_model = gensim.models.word2vec.Word2Vec.load(word2vec_model_path_zhwiki_update_814)
    all_file_modules = []
    length = len(file_paths)
    for i, file_path in enumerate(file_paths):
        print('number', i, '---total:', length)
        last_prex = file_path.split('.')[-1]
        if last_prex == 'docx':
            total_txt = get_docx_txt_with_format_list(file_path, title_list)
            modules = segment_one_resume_list_format(txt_list=total_txt,nn_model_w=nn_model_w, word2vec_model=word2vec_model)
            # print("******************************************************************************************")
            all_file_modules.append(modules)
    # for item in all_file_modules[0]:
    #     print(item[0])
    #     print(item[1])
    #     print('**********************************************************************************************')
    return all_file_modules


def get_files(train_path):
    file_names = os.listdir(train_path)
    # 一次性产生200个
    loc = 0
    while (loc + 200) <= len(file_names):
        yield [os.path.join(train_path, x) for x in file_names[loc:loc + 200]]
        loc += 200
    if loc < len(file_names):
        yield [os.path.join(train_path, x) for x in file_names[loc:len(file_names)]]
    # raise StopIteration


def segment_one_resume_from_file(file_path):
    """
    分割一份docx文档
    :param file_path: 文档所在路径, docx格式
    :return:按简历原始顺序排好的简历块列表
    """
    title_list = load_title_list(TITLE_LIST_PATH)
    txt_list = get_docx_txt_with_format_list(file_path, title_list)  # 得到带有格式信息的简历文本
    print(txt_list)
    modules = segment_one_resume_list_format_without_title(txt_list)
    # modules = [x[1] for x in modules]
    # return modules
    return modules


def pass_key(key_str, title_list):
    pat = '|'.join([key for key in title_list])
    res = re.search(pat, key_str)
    if res is not None and len(key_str) < 9:
        return True
    return False


def check_title_key(blocks, title_list):
    """
    如果检测发现该关键字非触发词，则合并相邻的两块
    :param blocks: 初步分块后的结果,[[key, info],[key, info],...]
    :param title_list: 标题关键字
    :return: 校验块触发词后的分块结果[label,label,...],[text,text,...]
    """
    temp = ['基本信息', '']
    while not pass_key(blocks[0][0], title_list):
        temp[1] += blocks[0][0] + blocks[0][1]
        blocks.remove(blocks[0])
    i = 1
    while i < len(blocks):
        if not pass_key(blocks[i][0], title_list):
            blocks[i-1][1] += blocks[i][0] + blocks[i][1]
            blocks.remove(blocks[i])
            i -= 1
        i += 1
    # 默认基本信息在最前
    if temp[1] != '':
        blocks.insert(0, temp)
    label_list = []
    info_list = []
    for item in blocks:
        label_list.append(item[0])
        info_list.append(item[1])
    return label_list, info_list


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


def segment_txt_list_to_module(txt_list, title_dic):
    """
    txt_list:文本集合
    title_dic:标签对应类别，字典，形如:{'基本信息':['base_info'],...}
    返回:[{},{},...]
    """
    pat = r'|'.join(title_dic.keys())
    res_list = []
    for item in txt_list:
        segment_res = segment_one_resume(item, pat)
        one_new_module_res = merge_to_new_module(segment_res=segment_res, title_dic=title_dic)
        res_list.append(one_new_module_res)
    return res_list


def load_sci_exp(path,segment):
    f = open(path, 'r', encoding='utf-8')
    txt = f.read()
    sci_exp_list = txt.split(segment)
    sci_exp_list = [(item, label_number['sci_exp']) for item in sci_exp_list]
    return sci_exp_list


def divide_file_by_date(dir_path, year=None, month= None, day=None):
    import time
    """
    返回最新修改日期（天）的文件
    :param dir_path:
    :return:
    """
    file_names = []
    max_day = 0
    files = os.listdir(dir_path)
    length = len(files)
    days = [0 for i in range(length)]
    for i, file in enumerate(files):
        file_name = os.path.join(dir_path, file)
        file_date = os.path.getctime(file_name)
        list1 = time.ctime(file_date).split(' ')
        day = int(list1[len(list1) - 3])
        if day >= max_day:
            days[i] = day
            max_day = day
    print('max_day:', max_day)
    for i, day in enumerate(days):
        if day == max_day:
            # print(day)
            file_name = os.path.join(dir_path, files[i])
            file_names.append(file_name)
    return file_names


def divide_file_by_date1(dir_path, year=None, month=None, day=None):
    import time
    """
    返回最新修改日期（天）的文件
    :param dir_path:
    :return:
    """
    file_names = []
    files = os.listdir(dir_path)
    localtime = time.localtime(time.time())
    if not year:
        year = localtime.tm_year
    if not month:
        month = localtime.tm_mon
    if not day:
        day = localtime.tm_mday
    for i, file in enumerate(files):
        file_name = os.path.join(dir_path, file)
        file_date = os.path.getctime(file_name)
        localtime = time.localtime(file_date)
        y = localtime.tm_year
        m = localtime.tm_mon
        d = localtime.tm_mday
        # list1 = time.ctime(file_date).split(' ')
        # d = int(list1[len(list1) - 3])
        # m = int(list1[len(list1) - 3])
        # y = int(list1[len(list1) - 3])
        if y > year:
            file_names.append(file_name)
        elif y == year:
            if m > month:
                file_names.append(file_name)
            elif m == month:
                if d >= day:
                    file_names.append(file_name)

    return file_names


if __name__ == '__main__':
    def test():
        load_school(ROOT_PATH + '\\material\\school1.txt')
        load_school(ROOT_PATH + '\\material\\school2.txt')
        load_school(ROOT_PATH + '\\material\\school3.txt')
        load_school(ROOT_PATH + '\\material\\school4.txt')
        load_school(ROOT_PATH + '\\material\\school5.txt')

        # txt_list, test_text_list = get_txt_list_from_file()
        # title_dic = load_title_list(TITLE_LIST_PATH)
        # new_module_res_list = segment_txt_list_to_module(txt_list=txt_list, title_dic=title_dic)
        # new_module_test_res_list = segment_txt_list_to_module(txt_list=test_text_list, title_dic=title_dic)
        count = 0
        # for i in range(len(new_module_res_list)):  # 真实文本数据train
        #     for item in new_module_res_list[i]:
        #         if len(new_module_res_list[i][item]) >= 4 and item == 'xxx':
        #             f = open(MUTI_CLASSFICATION_TRAIN_DATA_PATH + '\\' + str(count) + '_' + str(
        #                 label_number[str(item)]) + '.txt', 'w', encoding='utf-8')
        #             f.write(text_filter(new_module_res_list[i][item]))
        #             f.close()
        #             count += 1
        base_info_list = create_muti_classification_data(num=500, type='base_info')
        edu_back_list = create_muti_classification_data(num=500, type='edu_back')
        skill_list = create_muti_classification_data(num=500, type='skill')
        job_exp_list = create_muti_classification_data(num=500, type='job_exp')
        self_comment_list = create_muti_classification_data(num=500, type='self_comment')
        item_exp_list = create_muti_classification_data(num=20, type='item_exp')
        school_exp_list = create_muti_classification_data(num=500, type='school_exp')
        honour_list = create_muti_classification_data(num=500, type='honour')
        other_list = create_other_data()  # 431
        # other_list = []
        other_list = [(item, label_number['other']) for item in other_list]
        sci_exp_list = load_sci_exp(ROOT_PATH + '\\material\\patent.txt', '#')
        sci_exp_list.extend(load_sci_exp(ROOT_PATH + "\\material\\paper.txt", "#---#"))
        other_length = len(other_list)
        length = len(sci_exp_list)
        sci_exp_list = random.sample(sci_exp_list, length)
        generate_data_list = base_info_list[:len(base_info_list) * 4 // 5]
        generate_data_list.extend(edu_back_list[:len(edu_back_list) * 4 // 5])
        generate_data_list.extend(skill_list[:len(skill_list) * 4 // 5])
        generate_data_list.extend(job_exp_list[:len(job_exp_list) * 4 // 5])
        generate_data_list.extend(item_exp_list[:len(item_exp_list) * 4 // 5])
        generate_data_list.extend(self_comment_list[:len(self_comment_list) * 4 // 5])
        generate_data_list.extend(school_exp_list[:len(school_exp_list) * 4 // 5])
        generate_data_list.extend(honour_list[:len(honour_list) * 4 // 5])
        generate_data_list.extend(sci_exp_list[:length * 3 // 4])
        generate_data_list.extend(other_list[:other_length * 3 // 4])
        generate_data_list = random.sample(generate_data_list, len(generate_data_list))
        with open(MUTI_CLASSFICATION_TRAIN_DATA_PATH + '\\train_data.txt', 'w', encoding='utf-8') as f:
            for i, item in enumerate(generate_data_list):
                if len(item[0]) < 2:
                    continue
                if i != 0:
                    f.write('###' + str(item[1]))
                else:
                    f.write(str(item[1]))
                f.write('\n')
                f.write(item[0])
                f.write('\n')
                # f.close()
                count += 1
                print(count)
        count = 0
        f.close()
        print('begin to test data')
        # for i in range(len(new_module_test_res_list)):  # 真实文本数据test
        #     for item in new_module_test_res_list[i]:
        #         if len(new_module_test_res_list[i][item]) >= 4:
        #             f = open(MUTI_CLASSFICATION_TEST_DATA_PATH + '\\' + str(count) + '_' + str(
        #                 label_number[str(item)]) + '.txt', 'w', encoding='utf-8')
        #             f.write(text_filter(new_module_test_res_list[i][item]))
        #             f.close()
        #             count += 1
        generate_data_list = base_info_list[len(base_info_list) * 4 // 5:]
        generate_data_list.extend(edu_back_list[len(edu_back_list) * 4 // 5:])
        generate_data_list.extend(skill_list[len(skill_list) * 4 // 5:])
        generate_data_list.extend(job_exp_list[len(job_exp_list) * 4 // 5:])
        generate_data_list.extend(item_exp_list[len(item_exp_list) * 4 // 5:])
        generate_data_list.extend(self_comment_list[len(self_comment_list) * 4 // 5:])
        generate_data_list.extend(school_exp_list[len(school_exp_list) * 4 // 5:])
        generate_data_list.extend(honour_list[len(honour_list) * 4 // 5:])
        generate_data_list.extend(sci_exp_list[length * 3 // 4:])
        generate_data_list.extend(other_list[other_length * 3 // 4:])
        generate_data_list = random.sample(generate_data_list, len(generate_data_list))
        with open(MUTI_CLASSFICATION_TEST_DATA_PATH + '\\test_data.txt', 'w', encoding='utf-8') as f:
            for i, item in enumerate(generate_data_list):
                if len(item[0]) < 2:
                    continue
                if i != 0:
                    f.write('###' + str(item[1]))
                else:
                    f.write(str(item[1]))
                f.write('###' + str(item[1]))
                f.write('\n')
                f.write(item[0])
                f.write('\n')
                # f.close()
                count += 1
                print(count)
            # f.write('###')
        f.close()

    def test1():
        path = MATERIAL_TRAIN_PATH + '\\【大数据算法研究员 _ 杭州 20k-40k】白洋 13年.docx'
        resume_txt = get_one_resume_txt(path)
        pat = ''
        title_list = load_title_list(ROOT_PATH + '\\material\\title_list.txt')
        print(title_list)
        for title in title_list:
            pat += '|' + '\n' + title + '.{,25}\n'
        res = segment_one_resume1(resume_txt, pat[1:])
        for item in res:
            print('key:', item[0])
            print(item[1])
            print('-------------------------------------------------------------------------------------------------------')

        labels, texts = check_title_key(res, title_list)
        print(labels)
        for item in texts:
            print(item)
            print('******************************************************************************************************')


    def test_segment_one_resume():
        title_list = load_title_list(TITLE_LIST_PATH)
        file_path = 'D:\\Download\\简历模板.docx'  # 简历文档路径
        txt_list = get_docx_txt_with_format_list(file_path, title_list)  # 得到带有格式信息的简历文本
        modules = segment_one_resume_list_format(txt_list)
        for item in modules:
            print(item[0])
            print(item[1])
            print('***************************************************************************************************')

    def test_segment_one_resume1():
        modules = segment_one_resume_from_file('D:\\Download\\简历模板.docx')
        for module in modules:
            print(module)

    def test1_save_rnn_block_data():
        title_list = load_title_list(TITLE_LIST_PATH)
        txt_list = segment_all_files(MATERIAL_TRAIN_PATH, MATERIAL_TEST_PATH, title_list)
        txt_list1 = segment_all_files1(DOWNLOAD_RESUME_PATH, title_list)
        txt_list.extend(txt_list1)
        # txt_list:[ [[key,text],[key,text],...],  [[key,text],[key,text],...], ...  ]
        block_dic = {'base_info': [], 'edu_back': [], 'job_exp': [], 'item_exp': [], 'sci_exp': [], 'honour': [],
                     'skill': []
            , 'school_exp': [], 'self_comment': [], 'other': []}
        for resume in txt_list:
            for block in resume:  # block is [key,text]
                keys = block[0].split(' ')
                for key in keys:
                    block_dic[key].append(block[1])
        # print('key:', block_dic.keys())
        for key in block_dic:
            seg = '\n********************************************************************\n'
            txt = seg.join([x for x in block_dic[key] if len(x) >= 4])
            f = open(RNN_BLOCK_DATA_PATH + '\\' + key + '\\' + key + '.txt', 'w', encoding='utf-8')
            f.write(txt)
            f.close()


    def test1_save_rnn_block_data_update_by_day():
        title_list = load_title_list(TITLE_LIST_PATH)
        file_paths = divide_file_by_date1(DOWNLOAD_RESUME_PATH1, day=11)
        nn_model_w = load_models(muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec)
        nn_model_w.summary()
        word2vec_model = gensim.models.word2vec.Word2Vec.load(word2vec_model_path_zhwiki_update_814)
        txt_list = segment_all_files2(file_paths, title_list, nn_model_w=nn_model_w, word2vec_model=word2vec_model)
        # txt_list:[ [[key,text],[key,text],...],  [[key,text],[key,text],...], ...  ]
        block_dic = {'base_info': [], 'edu_back': [], 'job_exp': [], 'item_exp': [], 'sci_exp': [], 'honour': [],
                     'skill': []
            , 'school_exp': [], 'self_comment': [], 'other': []}
        for resume in txt_list:
            for block in resume:  # block is [key,text]
                keys = block[0].split(' ')
                for key in keys:
                    block_dic[key].append(block[1])
        # print('key:', block_dic.keys())
        for key in block_dic:
            if key != 'other' and key != 'school_exp':
                continue
            seg = '\n********************************************************************\n'
            txt = seg.join([x for x in block_dic[key] if len(x) >= 2])
            f = open(RNN_BLOCK_DATA_PATH + '\\' + key + '\\' + key + '.txt', 'a', encoding='utf-8')
            f.write(seg)
            f.write(txt)
            f.close()
            # print('--------------------------------------',key,'------------------------------------------')
            # print(txt)


    def test2_rnn_data_real():
        title_list = load_title_list(TITLE_LIST_PATH)
        txt_list = segment_all_files(MATERIAL_TRAIN_PATH, MATERIAL_TEST_PATH, title_list)
        # txt_list:[ [[key,text],[key,text],...],  [[key,text],[key,text],...], ...  ]
        max_modules = 0
        index = 0
        avl_count = 0
        for i, txt in enumerate(txt_list):
            avl_count += len(txt)
            if max_modules < len(txt):
                index = i
                max_modules = len(txt)
            label = []
            for block in txt:  # 一份简历里的一个内容块,block = [key,text],key包含多个标题，用空格连接成的字符串
                try:
                    label_key = block[0].split(' ')
                    score = 1.0 / len(label_key)
                    one_label = [0 for i in range(len(number2label))]
                    for lab in label_key:
                        one_label[label_number[lab]] = score
                    label.append(one_label)
                except KeyError:
                    print('not found key:', lab)
            # create data
            data = "\n###\n".join([one_block[0] + '\n' + one_block[1] for one_block in txt])  # 每一个简历块之间用"\n######\n"隔开
            # save data
            data_file_name = RNN_DATA_DATA_PATH1 + '\\' + str(i) + '.txt'
            f = open(data_file_name, 'w', encoding='utf-8')
            f.write(data)
            # save label
            label_file_name = RNN_DATA_LABEL_PATH1 + '\\' + str(i) + '.txt'
            f = open(label_file_name, 'w', encoding='utf-8')
            for item in label:
                f.write(' '.join([str(elem) for elem in item]))
                f.write('\n')
            f.close()
        avl_count /= len(txt_list)
        print('average module count is:', int(avl_count))
        print("max_modules:", max_modules)
        print("index is:", index)


    def test_segment_all_files():
        title = load_title_list(TITLE_LIST_PATH)
        segment_all_files(MATERIAL_TRAIN_PATH, MATERIAL_TEST_PATH, title)


    def test_segment_all_files_download():
        title = load_title_list(TITLE_LIST_PATH)
        segment_all_files1(DOWNLOAD_RESUME_PATH, title)


    def test_segment_all_files_by_iterator():
        iter_obj = get_files(DOWNLOAD_RESUME_PATH1)
        title_list = load_title_list(TITLE_LIST_PATH)
        for item in iter_obj:
            # segment_all_files2(item, title_list)
            print(len(item))

    def test_save_muti_classification_data():
        original_path = RNN_BLOCK_DATA_PATH + '\\sci_exp\\sci_exp.txt'
        goal_path = MUTI_CLASSFICATION_TRAIN_DATA_PATH + '\\train_data.txt'
        with open(original_path, 'r', encoding='utf-8') as f:
            txt = f.read()
        print(txt.replace(r'********************************************************************', '###7'))
    # test()
    # test2_rnn_data_real()
    # test1_save_rnn_block_data()
    # test_segment_one_resume()
    # test_segment_all_files()
    # test_segment_all_files_download()
    # test1_save_rnn_block_data_update_by_day()
    # test_segment_all_files_by_iterator()
    # test_save_muti_classification_data()
    test_segment_one_resume1()
