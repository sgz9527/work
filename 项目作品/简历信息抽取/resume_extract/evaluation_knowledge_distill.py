# coding=utf-8

"""
@File  : evaluation_mutual_learning_model.py
@Author: Xu Qiqiang
@Date  : 2020/12/9 0009
"""
# coding=utf-8

"""
@File  : evaluation_dynamic_decision_hybrid_model.py
@Author: Xu Qiqiang
@Date  : 2020/11/30 0030
"""
import tensorflow as tf
import random
from special_string import *
from static_num import *
from load_resume_data import load_data_for_rnn, load_data_for_rnn_new, load_data_for_rnn_new_add_noise
from feature_engineer import trans_to_wordvec_by_word2vec, process_rnn_label_list
from sklearn.metrics import recall_score, precision_score, f1_score
from resume_segement import segment_one_resume_from_file
from create_data import number2label
import numpy as np
from math import log # 自然对数
import scipy.stats
import math


class BRNNFeatureFusion(tf.keras.Model):
    def __init__(self, time_step, feature_size, rnn_utils, rnn_layers_num=1, T=4):
        super(BRNNFeatureFusion, self).__init__()
        self.T = T
        self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense = tf.keras.layers.Dense(units=11)
        self.time_step = time_step
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(0.1)
        self.feature_size = feature_size
        self.rnn_layers_num = rnn_layers_num
        self.rnn_utils_num = rnn_utils
        self.batch_size = None
        self.gru = []
        for i in range(rnn_layers_num):
            self.gru.append(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=rnn_utils, return_sequences=True, activation='relu')))

    def call(self, x, training=True):
        """

        :param x: (sample_num, time_step, item_d)
        :param training:
        :return:
        """
        x1 = x
        for i in range(self.rnn_layers_num):
            x1 = self.gru[i](x1)
        outputs = []
        # outputs_soft = []
        for i, item in enumerate(tf.unstack(x1, axis=1)):
            out_brnn = self.dense0(item)  # (sample_size, 64)  # rnn
            out_fnn = self.dense1(x[:, i, :])  # (sample_size, 64)  # fnn
            out_fnn = self.dense2(out_fnn)
            out = tf.concat([out_brnn, out_fnn], axis=1)  # (sample_size, 128)
            out = self.batch_normalization1(out)
            out1 = self.drop(out)
            out2 = self.dense3(out1)
            out2 = self.batch_normalization2(out2)
            out = tf.concat([out1, out2], axis=1)  # (sample_size, 256)
            out = self.dense(out)
            # out_soft = tf.nn.softmax(tf.divide(out, self.T))
            out = tf.nn.softmax(out)
            outputs.append(out)
            # outputs_soft.append(out_soft)

        # return tf.stack(outputs, axis=1), tf.stack(outputs_soft, axis=1)  # 最终结果shape:(batchsz, time_step, 11)
        return tf.stack(outputs, axis=1)


class FNNModel(tf.keras.Model):
    def __init__(self, time_step, feature_size, T=4):
        super(FNNModel, self).__init__()
        self.T = T
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu', name='dense1')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu', name='dense2')
        self.dense_fnn = tf.keras.layers.Dense(units=11, activation='softmax', name='dense_fnn')
        self.time_step = time_step
        self.feature_size = feature_size
        self.batch_size = None

    def call(self, x):
        """

        :param x: (sample_num, time_step, item_d)
        :param training:
        :return:
        """
        # print(x.shape)
        outputs_fnn = []
        for i, item in enumerate(tf.unstack(x, axis=1)):
            out_fnn = self.dense1(item)  # (sample_size, 64)  # fnn
            out_fnn = self.dense2(out_fnn)
            out_fnn = self.dense_fnn(out_fnn)
            outputs_fnn.append(out_fnn)
            # print('out_fnn.shape, final:', out_fnn.shape)
        return tf.stack(outputs_fnn, axis=1)  # 最终结果shape:(batchsz, time_step, 11)


class MyRecall(tf.keras.metrics.Metric):
    def __init__(self, name=None, classes=10):
        """
        召回率 :所有实际为正例的类别预测也为正例的数量占所有实际为正例的数量的比例
        :param name:
        :param classes:类别总数
        """
        super().__init__(name=name)
        self.true_positives, self.true_nagtives = [], []  # 总共有classes个类别，每个类别都对应一个预测为正例且真实为正确， 预测为正例且真实为错误的指标
        for i in range(classes):
            self.true_positives.append(self.add_weight(name='true_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))
            self.true_nagtives.append(self.add_weight(name='false_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))

    def update_state1(self, y_pred_val, y_true_val, sample_weight=None):
        """
        对于真实标签序列中one_hot全为0的空白标签截断，不参与评估
        :param y_true_val: 真实类别序列
        :param y_pred_val: 预测类别序列
        :param sample_weight:
        :return:
        """
        # print('after reshape,y_true.shape:', y_true.shape)
        # print('after reshape,y_pred.shape:', y_pred.shape)
        # print(y_true_argmax.shape)
        length = y_true_val.shape[0]
        # print('length is:', length)
        for i in range(length):
            if y_pred_val[i] == y_true_val[i]:
                self.true_positives[y_true_val[i]].assign_add(1)
            else:
                self.true_nagtives[y_true_val[i]].assign_add(1)

    def update_state(selfself, pre, act, sample_weight=None):
        pass

    def result(self):
        TP, TN = tf.stack(self.true_positives, axis=0), tf.stack(self.true_nagtives, axis=0)
        res = tf.reduce_mean(TP/(TP + TN))
        if math.isnan(res):
            return 0.0
        return res


class MyPrecision(tf.keras.metrics.Metric):
    def __init__(self, name=None, classes=10):
        """
        精确率 :所有预测为正例的类别实际也为正例的数量占所有预测为正例的数量的比例
        :param name:
        :param classes:类别总数
        """
        super().__init__(name=name)
        self.true_positives, self.false_positives = [], []  # 总共有classes个类别，每个类别都对应一个预测为正例且真实为正确， 预测为正例且真实为错误的指标
        for i in range(classes):
            self.true_positives.append(self.add_weight(name='true_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))
            self.false_positives.append(self.add_weight(name='false_positive' + str(i), dtype=tf.int32, initializer=tf.zeros_initializer()))

    def update_state1(self, y_pred_val, y_true_val, sample_weight=None):
        """
        对于真实标签序列中one_hot全为0的空白标签截断，不参与评估
        :param y_true: 真实类别序列 shape:(sample_size, time_step, one_hot_size)
        :param y_pred: 预测类别序列 shape:(sample_size, time_step, one_hot_size)
        :param sample_weight:
        :return:
        """
        # print('after reshape,y_true.shape:', y_true.shape)
        # print('after reshape,y_pred.shape:', y_pred.shape)
        # print(y_true_argmax.shape)
        length = y_true_val.shape[0]
        # print('length is:', length)
        for i in range(length):
            if y_pred_val[i] == y_true_val[i]:
                self.true_positives[y_pred_val[i]].assign_add(1)
            else:
                self.false_positives[y_pred_val[i]].assign_add(1)

    def update_state(selfself, pre, act, sample_weight=None):
        pass

    def result(self):
        TP, FP = tf.stack(self.true_positives, axis=0), tf.stack(self.false_positives, axis=0)
        res = tf.reduce_mean(TP/(TP + FP))
        if math.isnan(res):
            return 0.0
        return res


class MyAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        对于真实标签序列中one_hot全为0的空白标签截断，不参与评估
        :param y_true: 真实类别序列 shape:(sample_size, time_step, one_hot_size)
        :param y_pred: 预测类别序列 shape:(sample_size, time_step, one_hot_size)
        :param sample_weight:
        :return:
        """
        y_true = tf.reshape(y_true, [y_true.shape[0]*y_true.shape[1], y_true.shape[2]])
        y_pred = tf.reshape(y_pred, [y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]])
        # print('after reshape,y_true.shape:', y_true.shape)
        # print('after reshape,y_pred.shape:', y_pred.shape)
        y_true_argmax = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        # print('y_true_argmax.shape:', y_true_argmax.shape)

        values = tf.cast(tf.equal(y_true_argmax, y_pred_argmax), tf.int32)
        # print('values.shape:', values.shape)
        # self.total = self.total+tf.shape(y_true)[0]
        self.total.assign_add(y_true.shape[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total

    # def reset_states(self):
    #     self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
    #     self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())


class KnowledgeDistillModel(tf.keras.Model):
    def __init__(self, time_step, feature_size, rnn_utils, rnn_layers_num=1, fnn_model_path=None, T=5):
        super(KnowledgeDistillModel, self).__init__()
        self.time_step = time_step
        self.T = T
        self.batch_size = None
        self.brnn= BRNNFeatureFusion(time_step=time_step, feature_size=feature_size, rnn_utils=rnn_utils, rnn_layers_num=rnn_layers_num)
        self.fnn = FNNModel(time_step=time_step, feature_size=feature_size)
        self.fnn.load_weights(fnn_model_path)
        # print('len(fnn.layers):', self.fnn.layers)
        for i in range(len(self.fnn.layers)):
            self.fnn.layers[i].trainable = False

    @staticmethod
    def calc_ent(x):
        if np.argmax(x) == 10:
            return 0.0
        ent = 0.0
        for p in x:
            ent -= p * np.log(p)
        if math.isnan(ent):
            # print('exist nan', ent)
            ent = 0.0
        return ent

    @staticmethod
    def calc_ent1(x):
        ent = 0.0
        for row in x:
            ent += KnowledgeDistillModel.calc_ent(row)
        return ent

    def call(self, x):
        """

        :param x: (sample_num, time_step, item_d)
        :return:
        """
        out1 = self.brnn(x)  # (sample_size, time_step, 11)
        out2 = self.fnn(x)  # (sample_size, time_step, 11)

        return out1, out2  # 最终结果shape:(batchsz, time_step, 10)

    @staticmethod
    def cct(actual, pred):
        """
        计算N个样本的平均交叉熵
        :param actual: (batch_size, one_hot_size)
        :param pred: (batch_size, one_hot_size)
        :return:loss
        """
        # actual, pred = tf.unstack(actual, axis=0), tf.unstack(pred, axis=0)
        # length = len(actual)
        # cross_entropy = 0
        # for i in range(length):
        #     # if np.all(actual[i].numpy() == 0):
        #     # if actual[i].numpy()[-1] == 1:
        #     #     continue
        #     cross_entropy += -tf.reduce_sum(actual[i] * tf.math.log(tf.clip_by_value(pred[i], 1e-10, 1.0)))
        cross_entropy = tf.losses.categorical_crossentropy(actual, pred, label_smoothing=0.)
        return cross_entropy

    @staticmethod
    def cct_kl(p1, p2):
        """
        计算N个样本的平均交叉熵
        :param p1: (batch_size, one_hot_size)
        :param p2: (batch_size, one_hot_size)
        :return:loss
        """
        p1, p2 = tf.unstack(p1, axis=0), tf.unstack(p2, axis=0)
        length = len(p1)
        KL = 0
        for i in range(length):
            KL += scipy.stats.entropy(p1[i], p2[i])  # KL散度
        return KL

    @staticmethod
    def my_loss(actual, predict):
        # predict, actual shape:(batchsz, time_step,10)
        # print("actual is", actual)
        # print("predict is", predict)
        # actual = tf.one_hot(actual, depth=10)
        # print(actual.shape)
        total_loss = 0
        length = actual.shape[1]
        actual_items = tf.unstack(actual, axis=1)  # shape:(batchsz, 10)
        predict_items = tf.unstack(predict, axis=1)  # shape:(batchsz, 10)
        # cct = tf.keras.losses.CategoricalCrossentropy()
        for i in range(length):
            total_loss += KnowledgeDistillModel.cct(actual_items[i], predict_items[i])
        return total_loss

    @staticmethod
    def my_loss_kl(p1, p2):
        # predict, actual shape:(batchsz, time_step,10)
        # print("actual is", actual)
        # print("predict is", predict)
        # actual = tf.one_hot(actual, depth=10)
        # print(actual.shape)
        total_loss = 0
        length = p1.shape[1]
        actual_items = tf.unstack(p1, axis=1)  # shape:(batchsz, 10)
        predict_items = tf.unstack(p2, axis=1)  # shape:(batchsz, 10)
        # cct = tf.keras.losses.CategoricalCrossentropy()
        for i in range(length):
            total_loss += KnowledgeDistillModel.cct_kl(actual_items[i], predict_items[i])
        return total_loss

    def train_step(self, inputs, labels, optimizer, train_loss, train_precision, train_recall, lr=0.1, T=0.5, mode=1):  # 执行一个batch样本数量
        # labels.shape:(batch_size, time_step, 10)
        # print('inputs.shape:', inputs.shape)
        # print(inputs)
        if inputs.shape[0] < self.batch_size:
            print('not equal batch_size, concat to batch_size', inputs.shape, labels.shape)
            # 扩充至batch_size
            slices = tf.unstack(inputs, axis=0)  # 解堆叠，得到每一个样本
            slices_labels = tf.unstack(labels, axis=0)
            extra_index = [random.randint(0, len(slices) - 1) for i in range(self.batch_size - inputs.shape[0])]
            extra = []
            extra_labels = []
            length = self.batch_size - inputs.shape[0]
            for i in range(length):
                extra.append(slices[extra_index[i]])
                extra_labels.append(slices_labels[extra_index[i]])
            slices.extend(extra)  # 扩展至batch_size个
            slices_labels.extend(extra_labels)
            inputs = tf.stack(slices, axis=0)
            labels = tf.stack(slices_labels, axis=0)
            # extra_labels, extra_index, extra = None, None, None
            print(inputs.shape, labels.shape)

        with tf.GradientTape() as tape:
            predictions_brnn, predictions_fnn = self.call(inputs)
            # print(predictions_teacher)
            loss_label, loss_mutual = KnowledgeDistillModel.my_loss(labels, predictions_brnn), KnowledgeDistillModel.my_loss(predictions_fnn, predictions_brnn)
            # print('loss:', loss)
            loss = loss_label + T * loss_mutual
            gradients = tape.gradient(loss, self.trainable_variables)  # 反向求导
            for i, gradient in enumerate(gradients):
                gradients[i] = gradient * lr
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # 参数更新
        train_loss.update_state(loss)
        y_true = tf.reshape(labels, [labels.shape[0] * labels.shape[1], labels.shape[2]])
        y_pred = tf.reshape(predictions_brnn, [predictions_brnn.shape[0] * predictions_brnn.shape[1], predictions_brnn.shape[2]])

        # 去除真实标签onehot全为0的空白标记
        y_true, y_pred = tf.unstack(y_true, axis=0), tf.unstack(y_pred)
        for i, label in enumerate(y_true):
            # if np.all(label.numpy() == 0):
                # print('white space mark:', label.numpy())
            if label.numpy()[-1] == 1:
                y_true.pop(i)
                y_pred.pop(i)
        y_true, y_pred = tf.stack(y_true, axis=0), tf.stack(y_pred, axis=0)

        y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int32).numpy()
        y_true_argmax = tf.argmax(y_true, axis=-1, output_type=tf.int32).numpy()
        # print(y_pred_argmax)
        train_precision.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        train_recall.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        # print(labels, predictions)

        # train_loss(loss)
        # train_acc(labels, predictions)

    def fit(self, x_train, y_train, batchsz=16, epochs=5):
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        # train_acc = MyAccuracy(name='train_acc')
        train_precision = MyPrecision(name='precision', classes=11)
        train_recall = MyRecall(name='recall', classes=11)
        self.batch_size = batchsz
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batchsz)
        if x_train.shape[0] % batchsz == 0:
            batch_count = x_train.shape[0] // batchsz
        else:
            batch_count = x_train.shape[0] // batchsz + 1
        for epoch in range(epochs):
            print('--------------------Epoch', epoch+1, '/', epochs,'-----------------------')
            # 在下一次epoch开始时，重置评估指标
            train_loss.reset_states()
            # train_acc.reset_states()
            train_recall.reset_states()
            train_precision.reset_states()
            count = 0
            T = (0.45 / epochs) * epoch + 0.55
            print(T)
            if epoch >= 3*epochs/4:
                lr = 0.01
            else:
                lr = 0.05

            for inputs, labels in train_ds:  # 执行一个batch
                # print(inputs, labels)
                count += 1
                self.train_step(inputs, labels, optimizer, train_loss, train_precision, train_recall, lr, T=T)
                template = 'Batch {}/{}, Loss: {}, Precision:{}%, Recall:{}%'
                print(template.format(count, batch_count, train_loss.result(), train_precision.result(), train_recall.result()))
            print('')
            template = 'Epoch {}, Loss: {}, Precision:{}%, Recall:{}%'
            print(template.format(epoch + 1, train_loss.result(), train_precision.result(), train_recall.result()))
            for i in range(10):
                print('')

    def predict(self, inputs, number2label=None):
        """

        :param inputs: 输入样本数据 shape:(sample_size, time_step, feature_size)
        :param number2label: 索引对应标签名，若不为空，则返回具体类别名字 type:list
        :return:[[label1, label2,...], [label1, label2,...], ...]  label为字符串或者数字索引
        """
        outputs = self.call(inputs)[0]  # (sample_size, time_step, one_hot_size)
        y_pre_list = []
        if number2label is not None:
            if isinstance(number2label, list):
                samples = tf.unstack(outputs, axis=0)  # 按样本数量解开
                for sample in samples:
                    y_pre = tf.argmax(sample, axis=-1, output_type=tf.int32)
                    labels = []
                    for i in range(y_pre.shape[0]):
                        labels.append(number2label[tf.gather(y_pre, i).numpy()])
                    y_pre_list.append(labels)
            else:
                return None
        else:
            samples = tf.unstack(outputs, axis=0)  # 按样本数量解开
            for sample in samples:
                print(sample.numpy())
                y_pre = tf.argmax(sample, axis=-1, output_type=tf.int32)
                labels = []
                for i in range(y_pre.shape[0]):
                    labels.append(tf.gather(y_pre, i).numpy())
                y_pre_list.append(labels)
        return y_pre_list

    def evaluate(self, inputs, labels, choose=0):
        """
        评估测试集上的表现
        :param inputs:输入样本 (sample_size, time_step, feature_size)
        :param labels:(sample_size, time_step, one_hot_size)
        :return:返回test_loss, test_acc, test_precision, test_recall
        """
        # test_acc, test_precision, test_recall = MyAccuracy(name='test_acc'), MyPrecision(name='test_precision'),
        # MyRecall(name='test_recall')
        predictions = self.call(inputs)[choose]
        y_true = tf.reshape(labels, [labels.shape[0] * labels.shape[1], labels.shape[2]])
        y_pred = tf.reshape(predictions, [predictions.shape[0] * predictions.shape[1], predictions.shape[2]])
        # 去除真实标签onehot全为0的空白标记
        y_true, y_pred = tf.unstack(y_true, axis=0), tf.unstack(y_pred)
        for i, label in enumerate(y_true):
            # if np.all(label.numpy() == 0):
                # print('white space mark:', label.numpy())
            if label.numpy()[-1] == 1:
                y_true.pop(i)
                y_pred.pop(i)
        y_true, y_pred = tf.stack(y_true, axis=0), tf.stack(y_pred, axis=0)
        y_pred_argmax = tf.argmax(y_pred, axis=-1, output_type=tf.int32).numpy()
        y_true_argmax = tf.argmax(y_true, axis=-1, output_type=tf.int32).numpy()
        t_pre = precision_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        t_rec = recall_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        f1 = f1_score(y_true=y_true_argmax, y_pred=y_pred_argmax, average='weighted')
        # test_precision.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        # test_recall.update_state1(y_pred_val=y_pred_argmax, y_true_val=y_true_argmax)
        # t_acc, t_pre, t_rec = test_acc.result(), test_precision.result(), test_recall.result()
        return {'precision': t_pre, 'recall': t_rec, 'f1-score': f1}


if __name__ == '__main__':
    def evaluation_mbrnn():
        train_x, train_y, test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=700, test_num=400, noise_percent=10)
        brnn = KnowledgeDistillModel(time_step=TIME_STEP, feature_size=WORD2VEC_FEATURE_NUM, rnn_utils=RNN_UTILS,
                                     rnn_layers_num=RNN_LAYERS_NUM, fnn_model_path=FNN_700_PATH)
        begin = 0
        process_rnn_label_list(train_y, time_step=brnn.time_step, begin=begin)  # 原地修改label_list,统一维度
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        # print(train_y)
        train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=WORD2VEC_FEATURE_NUM,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn', time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=WORD2VEC_FEATURE_NUM,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn',
                time_step=brnn.time_step, begin=begin)
        train_x, train_y = tf.constant(train_x, dtype=tf.float32), tf.constant(train_y, dtype=tf.float32)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        inputs, label_list = None, None
        brnn.fit(train_x, train_y, batchsz=5, epochs=15)
        ev = brnn.evaluate(test_x, test_y)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))
        brnn.save_weights(KNOWLEDGE_DISTILL_MODEL_700_PATH)

    def evaluation_mbrnn1():
        train_x, train_y, test_x, test_y = load_data_for_rnn_new_add_noise(data_set=1, train_num=1, test_num=400)
        brnn = KnowledgeDistillModel(time_step=TIME_STEP, feature_size=WORD2VEC_FEATURE_NUM, rnn_utils=RNN_UTILS, rnn_layers_num=RNN_LAYERS_NUM, brnn_model_path=BBRNN_MODEL_PATH, fnn_model_path=FNN_MODEL_PATH)
        begin = 0
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=WORD2VEC_FEATURE_NUM,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn',
                time_step=brnn.time_step, begin=begin)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        inputs, label_list = None, None
        brnn.fit(test_x, test_y, batchsz=10, epochs=15)
        brnn.save_weights(HYBRID_MODEL_DYNAMIC_WEIGHT_PATH)

    def evaluation_mbrnn_best_params():
        test_xs, test_ys = [], []
        for i in range(0, 11):  # 异常比例从0到100%
            test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=1, test_num=400, noise_percent=i, load_train=False)
            test_xs.append(test_x)
            test_ys.append(test_y)
        begin = 0
        for i in range(len(test_ys)):
            process_rnn_label_list(test_ys[i], time_step=TIME_STEP, begin=begin)
            test_xs[i] = trans_to_wordvec_by_word2vec(test_xs[i], feature_size=WORD2VEC_FEATURE_NUM,
                                              word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn',
                                              time_step=TIME_STEP, begin=begin)
            test_xs[i], test_ys[i] = tf.constant(test_xs[i], dtype=tf.float32), tf.constant(test_ys[i], dtype=tf.float32)
        print('-----------------------------------------------')
        model = KnowledgeDistillModel(time_step=TIME_STEP, feature_size=WORD2VEC_FEATURE_NUM,
                                      rnn_utils=RNN_UTILS, rnn_layers_num=RNN_LAYERS_NUM)
        model.load_weights(KNOWLEDGE_DISTILL_MODEL_700_PATH)
        res = 0
        length = len(test_xs)
        for i, test_x in enumerate(test_xs):
            ev = model.evaluate(test_x, test_ys[i])
            print(i*10, '%:', ev['f1-score'])
            res += ev['f1-score']

        res /= length
        print('brnn mean f1-score:', res)
        res = 0
        for i, test_x in enumerate(test_xs):
            ev = model.evaluate(test_x, test_ys[i], choose=1)
            print(i*10, '%:', ev['f1-score'])
            res += ev['f1-score']

        res /= length
        print('fnn mean f1-score:', res)

    def evaluation_mbrnn_load_model():
        train_x, train_y, test_x, test_y = load_data_for_rnn_new_add_noise(data_set=3, train_num=1, test_num=400,
                                                                           noise_type='shuffle', noise_percent=10)
        brnn = KnowledgeDistillModel(time_step=TIME_STEP, feature_size=WORD2VEC_FEATURE_NUM, rnn_utils=RNN_UTILS,
                                     rnn_layers_num=RNN_LAYERS_NUM, fnn_model_path=FNN_700_PATH)
        brnn.load_weights(KNOWLEDGE_DISTILL_MODEL_700_PATH)
        begin = 0
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn',
                time_step=brnn.time_step, begin=begin)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        ev = brnn.evaluate(test_x, test_y)
        print('b-brnn:')
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))
        ev = brnn.evaluate(test_x, test_y, choose=1)
        print('fnn:')
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))

    def test_model():
        file_path = 'D:\\Download\\简历模板.docx'
        model_path = ROOT_PATH + '\\b_brnn_model_11-29'
        brnn_save = KnowledgeDistillModel(time_step=12, feature_size=100, rnn_utils=64, rnn_layers_num=2)
        brnn_save.load_weights(model_path)
        one_resume = segment_one_resume_from_file(file_path)
        for module in one_resume:
            print(module)
            print('-------------------------------------------------------------------------')
        text_list = [one_resume]
        inputs = trans_to_wordvec_by_word2vec(text_list, feature_size=100,
                                              word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn',
                                              time_step=brnn_save.time_step)

        print(brnn_save.predict(inputs), brnn_save.predict(inputs, number2label=number2label))

    # evaluation_mbrnn_load_model()
    # evaluation_mbrnn_best_params()
    evaluation_mbrnn()
    # test_model()




