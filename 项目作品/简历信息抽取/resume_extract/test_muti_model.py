import tensorflow.keras as kr
import tensorflow as tf
from keras.layers import Flatten,Dropout,BatchNormalization, Dense,MaxPooling1D,Conv1D,Input
from keras import Sequential
from special_string import *
from feature_engineer import trans_to_wordvec_by_tf_idf,trans_to_wordvec_by_word2vec
from resume_segement import number2label
from static_num import WORD2VEC_FEATURE_NUM,MAX_LEN
import joblib
from load_resume_data import get_docx_txt_with_formal_table


def create_model(max_len, word2vec_feature_size, output_size):
    model = Sequential()
    model.add(Input(shape=(max_len, word2vec_feature_size), name='input'))
    model.add(Conv1D(100, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Conv1D(100, 4, padding='same', activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Conv1D(100, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def load_models(file_path):
    return kr.models.load_model(file_path)


def load_text_cnn_model(file_path):
    model = create_model(max_len=MAX_LEN, word2vec_feature_size=WORD2VEC_FEATURE_NUM, output_size=10)
    model.load_weights(file_path, by_name=True)
    json_str = model.to_json()
    return kr.models.model_from_json(json_str)


def load_models1(file_path):
    return joblib.load(file_path)


if __name__ == '__main__':
    # nn_model = load_models(ROOT_PATH+'\\material\\ts_muti_nn_model3.h5')
    nn_model_w = load_models(muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec)
    # nn_model_w.build((None, 350, 100))
    nn_model_w.summary()
#
#     text_list = ['姓名：五五开，性别：男，年龄：23岁 地址：江西赣州 邮箱：45878@qq.com 微信：54564ajrg QQ:15678165',
#                  '姓名:司寇信敬 性别：男 所属民族:汉族 故乡:内蒙古 毕业学校：上海师范大学 政治面貌： 年龄：36 电话：18979293043 邮箱：BMzlE26SL@nuaa.com 社交账号：QQ:1588918744 ',
#                  '2013年9月02日-2017年05月01日  学历:小学生 所学专业:农业 ',
#                  '微信：奥IE网通几日 QQ：235465645，邮箱：349509@qq.com,github:https://www.github.com',
#                  '阿里巴巴，华为算法工程师工作2年以上',
#                  '2017-至今 人脸识别预警系统 项目介绍：该项目利用深度学习技术实现人脸识别功能 ',
# '王建洪. 中文电子病历信息提取方法研究[D].湖南大学,2016.[] 王若佳,魏思仪,王继民.BiLSTM-CRF模型在中文电子病历命名实体识别中的应用研究[J].文献与数据学报,2019,1(02):53-66.',
#                  '精通数据结构，熟练掌握机器学习，数据挖掘，大数据',
#                  '姓名：五五开，性别：男，年龄：23岁',
#                  '曾经参加过十佳歌手，并且组织过社团，担任学生会干部',
#                  '到岗时间可随时到岗工作性质：全职希望行业：计算机硬件目标地点：江苏期望月薪：面议/月目标职能：项目测试总监']
#     text_array = trans_to_wordvec_by_word2vec(text_list, feature_size=WORD2VEC_FEATURE_NUM, type='cnn', max_len=MAX_LEN, word2vec_model=word2vec_model_path_zhwiki_update_814)  # 转为词向量
#     res = nn_model_w.predict(text_array)
#     print('分类结果：', res)
#     for item in res:
#         print(number2label[list(item).index(max(item))])
    inputs = tf.zeros([1, 150, 100])
    print(nn_model_w(inputs))
