import tensorflow as tf
import random
from special_string import *
from load_resume_data import load_data_for_rnn, load_data_for_rnn_new
from feature_engineer import trans_to_wordvec_by_word2vec, process_rnn_label_list
from sklearn.metrics import recall_score, precision_score, f1_score
from create_data import number2label
from resume_segement import segment_one_resume_from_file
from evaluation_attention_bbrnn import PositionalEncoding
import numpy as np
import math

__all__ = [tf]


class MyDense(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(MyDense, self).__init__()
        self.n_outputs = n_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel',
                                        shape=[int(input_shape[-1]),
                                               self.n_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


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


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(AttentionLayer, self).__init__()
        self.n_outputs = n_outputs
        # self.weight_k, self.weight_q, self.weight_v, self.w = None, None, None, None
        self.softmax = tf.nn.softmax
        self.pos = PositionalEncoding()

    def build(self, input_shape):
        self.weight_k = self.add_weight('k',
                                          shape=[int(input_shape[-1]),
                                               self.n_outputs])
        self.weight_q = self.add_weight('q',
                                          shape=[int(input_shape[-1]),
                                                 self.n_outputs])
        self.weight_v = self.add_weight('v',
                                          shape=[int(input_shape[-1]),
                                                 self.n_outputs])

    def call(self, x):
        """
        input.shape:(sample_size, word_num, feature_size)
        :param x:
        :return:
        """
        x = self.pos(x)  # 加入位置编码
        # print('????')
        Query, K, V = [], [], []  # 所有样本的查询、键、值向量
        # 按时间步展开计算所有样本在每一步上的查询、键、值向量

        for item in tf.unstack(x, axis=1):
            # item.shape:(sample_size, feature_size)
            # print(item)
            # print(self.weight_k)
            # print('item.shape:', item.shape, 'self.weight_k.shape:', self.weight_k.shape)
            K.append(tf.matmul(item, self.weight_k))  # tf.matmul(item, self.weight_k)：(sample_size, n_outputs)
            V.append(tf.matmul(item, self.weight_v))  # tf.matmul(item, self.weight_v)：(sample_size, n_outputs)
            Query.append(tf.matmul(item, self.weight_q))  # tf.matmul(item, self.weight_Q)：(sample_size, n_outputs)
        # count = 0
        # print('x.shape:', x.shape)
        outputs = []
        V = tf.stack(V, axis=1)  # (sample_size, word_num, n_outputs)
        for Q in Query:
            # print('Q.shape:', Q.shape)
            # Q:(sample_size, n_outputs). 每一个样本只有一个查询向量，而每一个样本的键值向量则有word_num个
            # 注意力计算公式：softmax(q*K/(sqrt(d_k)) * V
            weights = []  # 存储所有时间步上的所有样本的注意力weight
            for one_step in K:  # 计算每一个时间步（单词）对分类的注意力权重
                # one_step.shape:(sample_size, n_outputs), tensor
                weight = tf.divide(tf.reduce_sum(tf.multiply(Q, one_step), axis=1), tf.sqrt(float(self.n_outputs)))
                # weight.shape:(sample_size, ) 所有样本在一个时间步上的权重
                # print('weight.shape:', weight.shape)
                weights.append(weight)

            weights = tf.stack(weights, axis=-1)  # (sample_size, word_num)  # 一行代表一个样本在所有步长上的权重
            # print('weights:', weights)
            weights = self.softmax(weights, axis=1)  # (sample_size, word_num)
            # print('after softmax:', weights)
            # print('V.shape:',V.shape)
            weights = tf.stack([weights], axis=-1)
            # print('before broadcast, weights.shape:', weights.shape)
            weights = tf.broadcast_to(weights, [V.shape[0], V.shape[1], V.shape[2]])
            # print('after broadcast, weights.shape:', weights.shape)
            output = tf.reduce_sum(tf.multiply(weights, V), axis=1)  # (sample_size, n_outputs)
            outputs.append(output)
            # print('final_output.shape:', final_output.shape)  # n个样本的最终表征向量

        return tf.stack(outputs, axis=1)


class GRUModel1(tf.keras.Model):
    def __init__(self, time_step, feature_size, rnn_utils, rnn_layers_num=1):
        super(GRUModel1, self).__init__()
        self.dense0 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=64, activation='relu')
        self.dense = tf.keras.layers.Dense(units=11, activation='softmax')
        self.time_step = time_step
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.drop = tf.keras.layers.Dropout(0.1)
        self.feature_size = feature_size
        self.rnn_layers_num = rnn_layers_num
        self.rnn_utils_num = rnn_utils
        self.batch_size = None
        # self.gru = []
        self.attentions = []
        self.batch_normalization = []
        # self.attention = AttentionLayer(64)
        for i in range(rnn_layers_num):
            # self.gru.append(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=rnn_utils, return_sequences=True, activation='relu')))
            self.attentions.append(AttentionLayer(64))
            self.batch_normalization.append(tf.keras.layers.BatchNormalization())

    def call(self, x, training=True):
        """

        :param x: (sample_num, time_step, item_d)
        :param training:
        :return:
        """
        # print(x.shape)
        x1 = x
        for i in range(self.rnn_layers_num):
            # x1 = self.gru[i](x1)
            x1 = self.attentions[i](x1)
            x1 = self.batch_normalization[i](x1)
        # print(x1 is x)
        outputs = []
        for i, item in enumerate(tf.unstack(x1, axis=1)):
            out1 = self.dense0(item)  # (sample_size, 64)
            # print('out1.shape:', out1.shape)
            out2 = self.dense1(x[:, i, :])  # (sample_size, 64)
            # print('out2.shape:', out2.shape)
            out = tf.concat([out1, out2], axis=1)  # (sample_size, 128)
            # print('out.shape:', out.shape)
            out = self.batch_normalization1(out)
            # print('out.shape, after batch normal:', out.shape)
            out = self.drop(out)
            # print('out.shape, after drop:', out.shape)
            # out = self.dense2(out)  # (sample_size, 64)
            # print('out.shape, after dense2:', out.shape)
            # out = self.batch_normalization2(out)
            # print('out.shape, after batch normal:', out.shape)
            out = self.dense(out)  # (sample_size, 11)最终输出
            # print('out.shape, final:', out.shape)
            outputs.append(out)
        return tf.stack(outputs, axis=1)  # 最终结果shape:(batchsz, time_step, 11)

    @staticmethod
    def cct(actual, pred):
        """
        计算N个样本的平均交叉熵
        :param actual: (batch_size, one_hot_size)
        :param pred: (batch_size, one_hot_size)
        :return:loss
        """
        actual, pred = tf.unstack(actual, axis=0), tf.unstack(pred, axis=0)
        length = len(actual)
        cross_entropy = 0
        for i in range(length):
            # if np.all(actual[i].numpy() == 0):
            # if actual[i].numpy()[-1] == 1:
            #     continue
            cross_entropy += -tf.reduce_sum(actual[i] * tf.math.log(tf.clip_by_value(pred[i], 1e-10, 1.0)))
        return cross_entropy

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
            total_loss += GRUModel1.cct(actual_items[i], predict_items[i])
        return total_loss

    def train_step(self, inputs, labels, optimizer, train_loss, train_precision, train_recall, lr=0.1):  # 执行一个batch样本数量
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
            predictions = self.call(inputs)
            loss = GRUModel1.my_loss(labels, predictions)
            # print('loss:', loss)
            # print(self.trainable_variables)
            gradients = tape.gradient(loss, self.trainable_variables)  # 反向求导
            for i, gradient in enumerate(gradients):
                gradients[i] = gradient * lr
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))  # 参数更新
        train_loss.update_state(loss)
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
            if epoch >= 2*epochs/3:
                lr = 0.01
            else:
                lr = 0.1
            for inputs, labels in train_ds:  # 执行一个batch
                # print(inputs, labels)
                self.train_step(inputs, labels, optimizer, train_loss, train_precision, train_recall, lr)
                count += 1
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
        outputs = self.call(inputs)  # (sample_size, time_step, one_hot_size)
        y_pre_list = []
        if number2label is not None:
            if isinstance(number2label, list):
                samples = tf.unstack(outputs, axis=0)  # 按样本数量解开
                for sample in samples:
                    y_pre = tf.argmax(sample, axis=-1, output_type=tf.int32)
                    # print('predict y is:', y_pre)
                    labels = []
                    for i in range(y_pre.shape[0]):
                        labels.append(number2label[tf.gather(y_pre, i).numpy()])
                    y_pre_list.append(labels)
            else:
                return None
        else:
            samples = tf.unstack(outputs, axis=0)  # 按样本数量解开
            for sample in samples:
                # print(sample.numpy())
                y_pre = tf.argmax(sample, axis=-1, output_type=tf.int32)
                labels = []
                for i in range(y_pre.shape[0]):
                    labels.append(tf.gather(y_pre, i).numpy())
                y_pre_list.append(labels)
        return y_pre_list

    def evaluate(self, inputs, labels):
        """
        评估测试集上的表现
        :param inputs:输入样本 (sample_size, time_step, feature_size)
        :param labels:(sample_size, time_step, one_hot_size)
        :return:返回test_loss, test_acc, test_precision, test_recall
        """
        # test_acc, test_precision, test_recall = MyAccuracy(name='test_acc'), MyPrecision(name='test_precision'),
        # MyRecall(name='test_recall')
        predictions = self.call(inputs)
        t_loss = GRUModel1.my_loss(labels, predictions)
        # test_acc.update_state(labels, predictions)
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
        train_x, train_y, test_x, test_y = load_data_for_rnn_new(data_set=3, train_num=500, test_num=400)
        brnn = GRUModel1(time_step=12, feature_size=100, rnn_utils=64, rnn_layers_num=2)
        begin = 0
        process_rnn_label_list(train_y, time_step=brnn.time_step, begin=begin)  # 原地修改label_list,统一维度
        process_rnn_label_list(test_y, time_step=brnn.time_step, begin=begin)
        # print(train_y)
        train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=100,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn', time_step=brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='rnn',
                time_step=brnn.time_step, begin=begin)
        train_x, train_y = tf.constant(train_x, dtype=tf.float32), tf.constant(train_y, dtype=tf.float32)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        inputs, label_list = None, None
        brnn.fit(train_x, train_y, batchsz=10, epochs=15)
        ev = brnn.evaluate(test_x, test_y)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))
        model_path = ROOT_PATH + '\\wbrnn_feature_block_attention_11_24_19_30'
        brnn.save_weights(model_path)

    def test_model():
        file_path = 'D:\\Download\\简历模板.docx'
        model_path = ROOT_PATH + '\\wbrnn_feature_block_attention_11_24_19_30'
        brnn_save = GRUModel1(time_step=12, feature_size=100, rnn_utils=64, rnn_layers_num=2)
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

    evaluation_mbrnn()
    test_model()

