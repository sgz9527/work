import tensorflow as tf
import keras

import numpy as np
# (batch_size,time_step,embedding_dim)
if __name__ == '__main__':

    def test1():
        batch_size = 3
        time_step = 2
        embedding_dim = 3
        train_x = tf.random.normal(shape=[batch_size,time_step,embedding_dim])
        hidden_dim = 4  # 隐藏层维度
        h0 = tf.random.normal(shape=[batch_size,hidden_dim])
        x0 = train_x[:,0,:]  # 第一个时间步的输入
        input = tf.constant(
        [[-0.03549285, -1.1304016, 0.39102417],
         [-0.56571656, 1.4268172,  0.07758367],
         [ 1.2091954,  0.54299754, 0.37603575]], dtype=float)
        state = tf.constant(
        [[-0.9344081,  1.3540316,  -0.12237156, 0.27811584],
         [1.6905733,  -0.8177901,  -0.26839492, -0.6950296],
         [0.41786417, -1.0966151, -2.1283705,  0.32684863]], dtype=float)

        # print(input)
        print(x0)
        print(state)
        print(h0)
        simpleRnnCell = tf.keras.layers.SimpleRNNCell(hidden_dim)
        out,h1 = simpleRnnCell(input, [state])  # 将当前时间步的x和上一时间步的隐藏层输出输入到
        print(out.shape,h1[0].shape)

    def test2():
        # 测试tf.one_hot
        import random
        label = np.zeros((3, 4))
        for i in range(label.shape[0]):
         for j in range(label.shape[1]):
          label[i, j] = random.randint(0, 9)
        label = tf.constant(label)
        label = tf.cast(label, tf.int32)
        print(label)
        label = tf.one_hot(label, 10)
        print(label)

    # test2()

    def test3():
        # 测试梯度更新
        @tf.function
        def cost_function(x):
            return x**2 + 8*x + 6

        @tf.function
        def get_grad(x):
            with tf.GradientTape() as tape:
                loss = cost_function(x[0])
                gradients = tape.gradient(loss, x)
            return gradients

        def train():
            lr = 0.1
            x = tf.Variable(0.3)
            y = tf.Variable(0.7)
            count = 0
            while count < 25:
                gr = get_grad([x, y])[0]
                # print(gr)
                x.assign_sub(lr * gr)
                count += 1
            print(x.numpy(), cost_function(x).numpy())

        train()
    test3()






