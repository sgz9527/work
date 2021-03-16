import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dropout,BatchNormalization, Dense,MaxPooling1D,Conv1D,Input
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from resume_extract.feature_engineer import change_to_tfidf, change_to_word2vec, trans_to_wordvec_by_word2vec
from load_resume_data import load_data, load_data_for_single_muti_classification
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from special_string import *
from static_num import WORD2VEC_FEATURE_NUM,MAX_LEN


def load_structure_data_by_tf_idf():
    """
    structure data
    """
    train_frame, test_frame = load_data()
    return change_to_tfidf(train_frame, test_frame, "muti_cv_model4.pkl")


def load_structure_data_by_word2vec(feature_num, word2vec_path, type='full', max_len=MAX_LEN):
    """
    structure data
    default value:type = full, represent full connect
    type = cnn,represent text cnn
    """
    print('train_model:load_structure_data_by_word2vec')
    train_frame, test_frame = load_data()
    return change_to_word2vec(train_frame, test_frame, feature_num, type=type, max_len=max_len,
                              word2vec_model=word2vec_path)


def load_structure_data_by_word2vec_new(feature_num, word2vec_path, type='cnn', max_len=MAX_LEN, data=1,
                                        train_size=50, test_size=400, val_size=100):
    train_x, train_y, test_x, test_y, val_x, val_y = load_data_for_single_muti_classification(data_set=data,
                                                            train_num=train_size, test_num=test_size, val_num=val_size)
    train_x = np.array(trans_to_wordvec_by_word2vec(train_x, feature_size=feature_num,
                                                    word2vec_model=word2vec_path, type=type, max_len=max_len))
    test_x = np.array(trans_to_wordvec_by_word2vec(test_x, feature_size=feature_num,
                                                   word2vec_model=word2vec_path, type=type, max_len=max_len))
    val_x = np.array(trans_to_wordvec_by_word2vec(val_x, feature_size=feature_num,
                                                    word2vec_model=word2vec_path, type=type, max_len=max_len))
    train_y, test_y, val_y = np.array(train_y), np.array(test_y), np.array(val_y)
    return train_x, train_y, test_x, test_y, val_x, val_y


def cross_validation(train_x, train_y, model):
    train_y = np.array(train_y)
    kf = KFold(n_splits=5)
    score_list = []
    for train_index,validate_index in kf.split(train_x):
        x_train, y_train = train_x[np.ix_(train_index)], train_y[np.ix_(train_index)]
        x_validate, y_validate = train_x[np.ix_(validate_index)], train_y[np.ix_(validate_index)]
        model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=0)
        scores = model.evaluate(x_validate, y_validate, verbose=0)
        score_list.append(scores[1])
    return np.mean(score_list)


def train_text_cnn(word2vec_feature_size, word2vec_path, max_len=MAX_LEN, output_size=10, data=1, train_num=50, test_num=400):
    """
    word2vec_feature_size:词嵌入向量的维数
    max_len:输入（句子）的最大长度，大于该长度则截断，小于该长度补0
    conv:卷积核的高度，相当于n-gram的大小,默认有4,5,6,7,8 这几种大小的卷积核
    output_size：神经网络最终输出
    """
    # 加载符合cnn输入的数据集
    # train_frame, test_frame = load_structure_data_by_word2vec(word2vec_feature_size, type='cnn',
    # max_len=max_len, word2vec_path=word2vec_path)
    train_x, train_y, test_x, test_y, val_x, val_y= load_structure_data_by_word2vec_new(word2vec_feature_size, type='cnn', max_len=max_len,
    word2vec_path=word2vec_path, data=data, train_size=train_num, test_size=test_num, val_size=100)
    filter_num = [50, 100, 150]
    filter_size = [[2, 3, 4],
                   [3, 4, 5],
                   [4, 5, 6]]
    drop1 = [0.1, 0.2, 0.3, 0.4]
    # train_x = np.zeros((train_frame['Text'].shape[0], max_len, word2vec_feature_size))
    # test_x = np.zeros((test_frame['Text'].shape[0], max_len, word2vec_feature_size))
    # for i, item in enumerate(train_frame['Text']):
    #     train_x[i] = item
    # for i, item in enumerate(test_frame['Text']):
    #     test_x[i] = item
    print('train_model:train_text_cnn:', train_x.shape)
    for filter_n in filter_num:
        for filter_s in filter_size:
            for drop1_p in drop1:
                model = Sequential()
                model.add(Input(shape=(max_len, word2vec_feature_size), name='input'))
                for item in filter_s:
                    model.add(Conv1D(filter_n, item, padding='same', activation='relu'))
                    model.add(MaxPooling1D(padding='same'))
                # model.add(Conv1D(100, 3, padding='same', activation='relu'))
                # model.add(MaxPooling1D(padding='same'))
                # model.add(Conv1D(100, 4, padding='same', activation='relu'))
                # model.add(MaxPooling1D(padding='same'))
                # model.add(Conv1D(100, 5, padding='same', activation='relu'))
                # model.add(MaxPooling1D(padding='same'))
                model.add(Flatten())
                model.add(Dropout(drop1_p))
                model.add(BatchNormalization())
                model.add(Dense(128, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(output_size, activation='softmax'))
                model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                # model.fit(train_x, train_frame['Label'], batch_size=128, epochs=15)
                # scores = model.evaluate(test_x, test_frame['Label'], verbose=0)
                print('entry cross_validation:')
                # score = cross_validation(train_x, train_frame['Label'], model)
                score = cross_validation(train_x, train_y, model)
                print(filter_n, filter_s, drop1_p, ':', score)


def save_train_text_cnn(word2vec_feature_size, word2vec_path, model_save_path, max_len=MAX_LEN, output_size=10, data=1, train_num=50, test_num=400):
    """
    word2vec_feature_size:词嵌入向量的维数
    max_len:输入（句子）的最大长度，大于该长度则截断，小于该长度补0
    conv:卷积核的高度，相当于n-gram的大小,默认有4,5,6,7,8 这几种大小的卷积核
    output_size：神经网络最终输出
    """
    # 加载符合cnn输入的数据集
    # train_frame, test_frame = load_structure_data_by_word2vec(word2vec_feature_size, type='cnn',
    # max_len=max_len, word2vec_path=word2vec_path)
    train_x, train_y, test_x, test_y, val_x, val_y = load_structure_data_by_word2vec_new(word2vec_feature_size,
                                                                           type='cnn',
                                                                           max_len=max_len,
                                                                           word2vec_path=word2vec_path, data=data,
                                                                           train_size=train_num, test_size=test_num, val_size=100)
    print('train_model:train_text_cnn:', train_x.shape)
    input_tensor = Input(shape=(max_len, word2vec_feature_size), name='input')
    print('after input_tensor')
    cnn1 = Conv1D(100, 3, padding='same', activation='relu')(input_tensor)
    print('after cnn1')
    cnn1 = MaxPooling1D(padding='same')(cnn1)
    cnn2 = Conv1D(100, 4, padding='same', activation='relu')(input_tensor)
    cnn2 = MaxPooling1D(padding='same')(cnn2)
    cnn3 = Conv1D(100, 5, padding='same', activation='relu')(input_tensor)
    cnn3 = MaxPooling1D(padding='same')(cnn3)
    print('after cnn3')
    cnn = tf.concat([cnn1, cnn2, cnn3], axis=-1)
    flat = Flatten()(cnn)
    # drop = Dropout(0.1)(flat)
    full = BatchNormalization()(flat)
    dense = Dense(128, activation='relu')(full)
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    # dense = Dropout(0.1)(dense)
    # dense = Dropout(0.1)(full1)
    output = Dense(output_size, activation='softmax')(dense)
    model = Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, tf.convert_to_tensor(train_y), batch_size=32, epochs=10)
    scores = model.evaluate(test_x, tf.convert_to_tensor(test_y), verbose=0)
    pres = model.predict(test_x)
    # print('pres.shape:', pres.shape)
    pres = np.argmax(pres, axis=1)
    print('pres.shape:', pres.shape)
    P = precision_score(test_y, pres, average='weighted')
    R = recall_score(test_y, pres, average='weighted')
    f1 = f1_score(test_y, pres, average='weighted')
    print('test data score:', scores)
    print('test data precision , recall, f1-score:', P, R, f1)
    # model.save(model_save_path)


def train(mode='tfidf', feature_num=None, word2vec_path=''):  # 采用全连接神经网路
    """
    mode:tfidf or word2vec
    """
    print('train_model:train')
    if mode == 'word2vec' and word2vec_path == '':
        print('word2vec path is null')
        return
    train_frame, test_frame = load_structure_data_by_word2vec(feature_num, type='full', word2vec_path=word2vec_path)
    train_x, train_y = train_frame['Text'], list(train_frame['Label'])
    test_x, test_y = test_frame['Text'], list(test_frame['Label'])
    train_x = np.array([[x for x in y] for y in train_x])
    test_x = np.array([[x for x in y] for y in test_x])
    """
    if mode =='tfidf':
        params = [32, 64, 80, 96, 112, 128]
        for param in params:
            inputs = tf.keras.Input(shape=(train_x.shape[1],), name='input')
            h1 = tf.keras.layers.Dense(192, activation='relu')(inputs)
            h1 = tf.keras.layers.Dense(128, activation='relu')(h1)
            outputs = tf.keras.layers.Dense(10, activation='softmax')(h1)
            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            score = cross_validation(train_x, train_y, model)
            print(param, ":", score)
    else:
        # 80,80,64 0.894
        params1 = [32, 48, 64, 80, 96, 112, 128, 144, 160]
        params2 = [16,32,48,64,80,96]
        for param1 in params1:
            for param2 in params2:
                inputs = tf.keras.Input(shape=(train_x.shape[1],), name='input')
                h1 = tf.keras.layers.Dense(param1, activation='relu')(inputs)
                h1 = tf.keras.layers.Dense(param2, activation='relu')(h1)
                outputs = tf.keras.layers.Dense(10, activation='softmax')(h1)
                model = tf.keras.Model(inputs, outputs)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                score = cross_validation(train_x, train_y, model)
                print((param1, param2), ":", score)

"""

    # print(train_x.shape[1])

    inputs = Input(shape=(train_x.shape[1],), name='input')
    if mode == 'tf-idf':
        h1 = Dense(192, activation='relu')(inputs)
        h1 = Dense(128, activation='relu')(h1)
    else:
        h1 = Dense(80, activation='relu')(inputs)
        h1 = Dense(80, activation='relu')(h1)
        h1 = Dense(64, activation='relu')(h1)
    outputs = Dense(10, activation='softmax')(h1)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=64, epochs=6, verbose=1)
    scores = model.evaluate(test_x, test_y, verbose=0)
    print(mode, ':', scores[1])
    # score = model.score(test_x, test_y)
    # print("神经网络得分:", cross_score.mean())
    # joblib.dump(model, ROOT_PATH + '\\material\\ts_muti_nn_model.pkl')
    if mode =='tfidf':
        model.save(muti_full_tfidf_model_path)
    else:
        model.save(muti_full_word2vec_model_path)
    print('save model sucessfully')


if __name__ == '__main__':
    # print(train_frame)
    # print(test_frame)
    save_train_text_cnn(word2vec_feature_size=WORD2VEC_FEATURE_NUM, max_len=MAX_LEN,
                        word2vec_path=word2vec_model_path_zhwiki_rnn_update_20_923,
                        model_save_path=muti_textcnn_api_model_update2_path_zhwiki_corpus_word2vec,
                        data=3, train_num=100, test_num=400)
    # 100, [3,4,5], 0.1, 128, 0.2
