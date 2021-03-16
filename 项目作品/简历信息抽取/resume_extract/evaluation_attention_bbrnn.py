import random
from special_string import *
from load_resume_data import load_data_for_rnn, load_data_for_rnn_new
from feature_engineer import trans_to_wordvec_by_word2vec, process_rnn_label_list
from sklearn.metrics import recall_score, precision_score, f1_score
from my_metric import *
from static_num import *
import numpy as np


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, mode='sum', **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        self.mode = mode
        super(PositionalEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.sequence_len = int(input_shape[-2])

    def call(self, x):
        """
        (sample_num, time_step, word_num, feature_size)
        :param x:
        :return:
        """
        if (self.embedding_dim == None) or (self.mode == 'sum'):
            self.embedding_dim = int(x.shape[-1])

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1

        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        if self.mode == 'sum':
            return position_embedding + x

        elif self.mode == 'concat':
            position_embedding = tf.reshape(
                tf.tile(position_embedding, (int(x.shape[0]), 1)),
                (-1, self.sequence_len, self.embedding_dim)
            )

            return tf.concat([position_embedding, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape

        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.embedding_dim)


class MyEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, n_outputs):
        super(MyEmbeddingLayer, self).__init__()
        self.n_outputs = n_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel',
                                        shape=[int(input_shape[-1]),
                                               self.n_outputs])

    def call(self, input):
        return tf.matmul(input, self.kernel)


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
        # 用于得到查询值
        self.w = self.add_weight('w', shape=[int(input_shape[-2]), 1])  # (word_num, 1)

    def call(self, x):
        """
        input.shape:(sample_size, word_num, feature_size)
        :param x:
        :return:
        """
        x = self.pos(x)  # 加入位置编码
        # print('????')
        Q, K, V = [], [], []  # 所有样本的查询、键、值向量
        # 按时间步展开计算所有样本在每一步上的键值向量
        for item in tf.unstack(x, axis=1):
            # item.shape:(sample_size, feature_size)
            # print(item)
            # print(self.weight_k)
            # print('item.shape:', item.shape, 'self.weight_k.shape:', self.weight_k.shape)
            K.append(tf.matmul(item, self.weight_k))  # tf.matmul(item, self.weight_k)：(sample_size, n_outputs)
            V.append(tf.matmul(item, self.weight_v))  # tf.matmul(item, self.weight_v)：(sample_size, n_outputs)
        # 按样本展开计算每一个样本的查询向量
        # count = 0
        # print('x.shape:', x.shape)
        for one_sample in tf.unstack(x, axis=0):
            # print(count)
            # count += 1
            # one_sample.shape:(word_num, feature_size)
            t = tf.matmul(tf.transpose(one_sample), self.w)  # (feature_size, 1) 列向量
            # print('t.shape:', t.shape)
            # print(self.weight_q.shape)
            q = tf.matmul(tf.transpose(t), self.weight_q)  # (1, n_outputs) 行向量
            Q.append(q)
        Q = tf.concat(Q, axis=0)
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
        V = tf.stack(V, axis=1)  # (sample_size, word_num, n_outputs)
        # print('V.shape:',V.shape)
        weights = tf.stack([weights], axis=-1)
        # print('before broadcast, weights.shape:', weights.shape)
        weights = tf.broadcast_to(weights, [V.shape[0], V.shape[1], V.shape[2]])
        # print('after broadcast, weights.shape:', weights.shape)
        final_output = tf.reduce_sum(tf.multiply(weights, V), axis=1)  # (sample_size, n_outputs)
        # print('final_output.shape:', final_output.shape)  # n个样本的最终表征向量

        return final_output


class WBRNNLayer(tf.keras.layers.Layer):
    """
    经过该层后，每个模块对应一个特征向量
    """
    def __init__(self, rnn_utils, output_vector_size):
        super(WBRNNLayer, self).__init__()
        self.GRU = tf.keras.layers.GRU(units=rnn_utils, return_sequences=False, activation='relu')
        self.attention = AttentionLayer(output_vector_size)

    def call(self, input):
        """
        input.shape:(sample_size, time_step, word_num, feature_size)
        :param input:
        :return:(sample_size, time_step, output_vector_size)
        """
        # time_step, word_num, feature_size = input.shape[0], input.shape[1], input.shape[2]
        block_list = tf.unstack(input, axis=1)
        output = []
        for block in block_list:  # block.shape:(sample_size, word_num, feature_size)
            # hidden_output_wbrnn = self.GRU(block)
            # output.append(hidden_output_wbrnn)
            # output.append(self.attention(hidden_output_wbrnn))
            output.append(self.attention(block))
        return tf.stack(output, axis=1)  # (sample_size, time_step, output_vector_size)


class HBRNN(tf.keras.Model):
    def __init__(self, time_step, feature_size, rnn_utils, word_num, rnn_layers_num=1, hidden_vector_size=64):
        super(HBRNN, self).__init__()
        self.dense = tf.keras.layers.Dense(units=11, activation='softmax')
        self.time_step = time_step
        self.feature_size = feature_size
        self.rnn_layers_num = rnn_layers_num
        self.rnn_utils_num = rnn_utils
        self.batch_size = None
        self.word_num = word_num
        self.gru = []
        self.wbrnn = WBRNNLayer(rnn_utils=rnn_utils, output_vector_size=hidden_vector_size)
        for i in range(rnn_layers_num):
            self.gru.append(tf.keras.layers.GRU(units=rnn_utils, return_sequences=True, activation='relu'))

    def call(self, x, training=True):
        """

        :param x: (sample_num, time_step, word_num, feature_size)
        :param training:
        :return:
        """
        # print(x.shape)
        x = self.wbrnn(x)  # trans to (sample_num, time_step, hidden_vector_size)
        for i in range(self.rnn_layers_num):
            x = self.gru[i](x)
        outputs = []
        for i, item in enumerate(tf.unstack(x, axis=1)):
            out = self.dense(item)
            outputs.append(out)
        return tf.stack(outputs, axis=1)  # 最终结果shape:(batchsz, time_step, 10)

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
            total_loss += HBRNN.cct(actual_items[i], predict_items[i])
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
            loss = HBRNN.my_loss(labels, predictions)
            # print('loss:', loss)
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
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale='dynamic')
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
            if epoch >= 3*epochs/4:
                lr = 0.02
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
        t_loss = HBRNN.my_loss(labels, predictions)
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


def set_gpu():
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print('true')
        gpu0 = gpus[0]  # 如果有多个GPU，仅使用第0个GPU
        # tf.config.experimental.set_memory_growth(gpu0, True)  # 设置GPU显存用量按需使用
        # 或者也可以设置GPU显存为固定使用量(例如：4G)
        # tf.config.experimental.set_virtual_device_configuration(gpu0,
        #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        # tf.config.set_visible_devices([gpu0], "GPU")
        # 比较GPU和CPU的计算速度

    # printbar()
    # with tf.device("/gpu:0"):
    #     tf.random.set_seed(0)
    #     a = tf.random.uniform((10000, 100), minval=0, maxval=3.0)
    #     b = tf.random.uniform((100, 100000), minval=0, maxval=3.0)
    #     c = a @ b
    #     tf.print(tf.reduce_sum(tf.reduce_sum(c, axis=0), axis=0))
    # printbar()
    #
    # printbar()
    # with tf.device("/cpu:0"):
    #     tf.random.set_seed(0)
    #     a = tf.random.uniform((10000, 100), minval=0, maxval=3.0)
    #     b = tf.random.uniform((100, 100000), minval=0, maxval=3.0)
    #     c = a @ b
    #     tf.print(tf.reduce_sum(tf.reduce_sum(c, axis=0), axis=0))
    # printbar()


if __name__ == '__main__':
    def evaluation_hbrnn():
        # from tensorflow.python.client import device_lib
        # print(device_lib.list_local_devices())
        # set_gpu()
        train_x, train_y, test_x, test_y = load_data_for_rnn_new(data_set=3, train_num=50, test_num=400)
        h_brnn = HBRNN(time_step=12, feature_size=100, rnn_utils=64, rnn_layers_num=1, hidden_vector_size=64, word_num=MAX_LEN)
        begin = 0
        process_rnn_label_list(train_y, time_step=h_brnn.time_step, begin=begin)  # 原地修改label_list,统一维度
        process_rnn_label_list(test_y, time_step=h_brnn.time_step, begin=begin)
        # print(train_y)
        train_x = trans_to_wordvec_by_word2vec(train_x, feature_size=100,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='attention', time_step=h_brnn.time_step, begin=begin)
        test_x = trans_to_wordvec_by_word2vec(test_x, feature_size=100,
                word2vec_model=word2vec_model_path_zhwiki_rnn_update_20_923, type='attention',
                time_step=h_brnn.time_step, begin=begin)
        train_x, train_y = tf.constant(train_x, dtype=tf.float32), tf.constant(train_y, dtype=tf.float32)
        test_x, test_y = tf.constant(test_x, dtype=tf.float32), tf.constant(test_y, dtype=tf.float32)
        print(train_x.shape, train_y.shape)
        print(test_x.shape, test_y.shape)
        inputs, label_list = None, None
        # wbrnn = WBRNNLayer(rnn_utils=32, output_vector_size=32)
        # output_train, output_test = wbrnn(train_x), wbrnn(test_x)
        # print(output_train.shape, output_test.shape)
        h_brnn.fit(train_x, train_y, batchsz=10, epochs=15)
        ev = h_brnn.evaluate(test_x, test_y)
        template = 'test data precision:{}, recall:{}, f1-score:{}'
        print(template.format(ev['precision'], ev['recall'], ev['f1-score']))


    def test_attention():
        # eg:(2,2,3)
        inputs = tf.Variable(
        [
            [[1, 1, 1],
             [2, 1, 2]],

        ], dtype=tf.float32)
        print(inputs)
        attention = AttentionLayer(4)
        output = attention(inputs)
        print(output)

    def test_wbrnn():
        # eg:(2,2,2,3)
        inputs = tf.Variable(
        [
            [
                [[1, 1, 1],
                [2, 1, 2]],

                [[1, 3, 5],
                [4, 2, 8]],
            ],
            [
                [[1, 7, 1],
                 [7, 3.5, 9]],

                [[1.2, 1.6, 0.4],
                 [0, 1.5, 3.2]],
            ]
        ], dtype=tf.float32)
        print(inputs)
        wbrnn = WBRNNLayer(12, 4)
        output = wbrnn(inputs)
        print(output)

    # test_wbrnn()
    # test_attention()
    evaluation_hbrnn()



