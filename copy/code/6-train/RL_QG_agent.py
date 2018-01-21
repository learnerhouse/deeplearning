import tensorflow as tf
import os
import gym
import numpy as np

class CNN:
    '''
        CNN
        conv4
    '''
    def __init__(self, env):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=1e-3)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(1e-3, shape=shape)
            return tf.Variable(initial)

        def conv2d(input_tensor, W):
            return tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding='SAME')

        def use_relu(conv, conv_biases):
            return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

        self.input_length = 8 * 8 * 3
        self.input_s = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.input_1 = tf.reshape(self.input_s, shape=[1, 8, 8, 3])
        # layer1-conv64
        self.W_conv_1 = weight_variable([3, 3, 3, 64])
        self.b_1 = bias_variable([64])
        self.conv_1 = conv2d(self.input_1, self.W_conv_1)
        self.out_1 = use_relu(self.conv_1, self.b_1)

        # layer2-conv64
        self.W_conv_2 = weight_variable([3, 3, 64, 64])
        self.b_2 = bias_variable([64])
        self.conv_2 = conv2d(self.out_1, self.W_conv_2)
        self.out_2 = use_relu(self.conv_2, self.b_2)

        # layer3-conv128
        self.W_conv_3 = weight_variable([3, 3, 64, 128])
        self.b_3 = bias_variable([128])
        self.conv_3 = conv2d(self.out_2, self.W_conv_3)
        self.out_3 = use_relu(self.conv_3, self.b_3)

        # layer4-conv128
        self.W_conv_4 = weight_variable([3, 3, 128, 128])
        self.b_4 = bias_variable([128])
        self.conv_4 = conv2d(self.out_3, self.W_conv_4)
        self.out_4 = use_relu(self.conv_4, self.b_4)

        # layer5-fc128
        self.out_4_flat = tf.reshape(self.out_4, [-1, 8 * 8 * 128])
        self.W_5 = weight_variable([8 * 8 * 128, 128])
        self.out_5 = tf.matmul(self.out_4_flat, self.W_5)

        # layer6-fc60
        self.W_6 = weight_variable([128, env.action_space.n])
        self.Q = tf.matmul(self.out_5, self.W_6)

        self.Q_target = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q))
        self.update = tf.train.GradientDescentOptimizer(1e-2).minimize(self.loss)
        self.init = tf.global_variables_initializer()

class Simple_model:
    '''
        a simple neural network
        use flatten input
    '''
    def __init__(self, env):
        self.input_length = env.board_size ** 2 * 3
        self.input_s = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform(shape=[self.input_length, env.action_space.n], minval=-1e-4, maxval=1e-4), name = 'w')
        # self.Q = tf.matmul(self.input_s, self.W)
        self.b1 = tf.Variable(tf.zeros([1, env.action_space.n]) + 1e-4)
        self.Q = tf.nn.relu(tf.matmul(self.input_s, self.W) + self.b1)
        # self.predict_action = tf.argmax(self.Q, 1)

        self.Q_target = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q))
        self.update = tf.train.GradientDescentOptimizer(1e-3).minimize(self.loss)
        self.init = tf.global_variables_initializer()

class RL_QG_agent:
    def __init__(self):
        # Init memory
        self.memory_size = 500
        self.memory_cnt = 0
        self.memory = []

        self.loading_model = True
        # self.loading_model = False
        self.model_type = 'simple_model'
        # self.model_type = 'cnn'

        self.eps = 0.5
        self.eps_min = 0.01
        self.eps_decay = 0.9999
        self.gamma = 0.99
        self.pre_train_step = 500
        self.update_freq = 5
        self.batch_size = 32

        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi_model")
        self.env = gym.make('Reversi8x8-v0')

        self.sess = tf.Session()
        self.init_model()

    def init_model(self):
        print(self.model_type)

        if self.model_type == 'simple_model':
            self.model = Simple_model(env = self.env)
            if not self.loading_model:
                # self.simple_train()
                self.train()
        else:
            self.model = CNN(env = self.env)
            if not self.loading_model:
                self.simple_train()
                # self.train()

        self.saver = tf.train.Saver()
        self.load_model()

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def remember(self, s, a, r, next_s, done, player):
        index = self.memory_cnt % self.memory_size
        if self.memory_cnt < self.memory_size:
            self.memory.append((s, a, r, next_s, done, player))
        else:
            self.memory[index] = (s, a, r, next_s, done, player)
        self.memory_cnt += 1

    def choose_action(self, state, enables, step):
        if np.random.rand(1) < self.eps or step < self.pre_train_step:
            return np.random.choice(enables)
        else:
            Q = self.sess.run(self.model.Q, \
            feed_dict={self.model.input_s: np.reshape(state, (1, self.model.input_length))})
            Q = np.ravel(Q)
            return enables[np.argmax(Q[enables])]

    def learn(self):
        upper_bound = min(self.memory_cnt, self.memory_size)
        batches = np.random.choice(upper_bound, self.batch_size)
        for i in batches:
            s, a, r, next_s, done, player = self.memory[i]
            Q = self.sess.run(self.model.Q, \
                              feed_dict={self.model.input_s: np.reshape(s, (1, self.model.input_length))})

            if done:
                max_next_Q = 0
            else:
                next_Q = self.sess.run(self.model.Q, \
                                  feed_dict={self.model.input_s: np.reshape(next_s, (1, self.model.input_length))})
                next_Q_flatted = np.ravel(next_Q)
                next_enables = self.env.get_possible_actions(next_s, 1 - player)
                max_next_Q = np.max(next_Q_flatted[next_enables])

            Q_target = Q
            Q_target[0][a] = r + self.gamma * max_next_Q

            # update
            update = self.sess.run(self.model.update, \
                         feed_dict={self.model.Q_target: Q_target, self.model.input_s: np.reshape(s, (1, self.model.input_length))})

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def __del__(self):
        '''
        好像没有用啊
        :return:
        '''
        print('sess close')
        self.sess.close()

    def place(self,state,enables,player):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        Q = self.sess.run(self.model.Q, \
                            feed_dict={self.model.input_s: np.reshape(state, (1, self.model.input_length))})
        Q = np.ravel(Q)
        action = enables[np.argmax(Q[enables])]
        return action

    def train(self):
        self.sess.run(self.model.init)
        play_game_times = 1000
        step = 0
        for i in range(play_game_times):
            s = self.env.reset()
            player = 0
            step_in_a_game = 0
            while True:
                step_in_a_game += 1
                enables = self.env.possible_actions
                a = self.choose_action(s, enables, step)
                next_s, r, done, _ = self.env.step((a, player))

                self.remember(s, a, r, next_s, done, player)

                if (step > self.pre_train_step) and (step % self.update_freq == 0):
                    self.learn()

                if done:
                    print('game {} : {}'.format(i, step_in_a_game))
                    break
                s = next_s
                player ^= 1
                step += 1


    def simple_train(self):
        # Start training
        eps = 1
        gamma = 0.99
        play_game_times = 1000
        eps_min = 0.01
        eps_decay = 0.999

        with tf.Session() as sess:
            sess.run(self.model.init)
            for i in range(play_game_times):
                s = self.env.reset()
                step = 0
                player = 0
                while True:
                    step += 1
                    # get Q
                    Q = sess.run(self.model.Q,\
                                    feed_dict={self.model.input_s: np.reshape(s, (1, self.model.input_length))})

                    enables = self.env.possible_actions
                    # get next action
                    if np.random.rand(1) < eps:
                        a = np.random.choice(enables)
                    else:
                        Q_flatted = np.ravel(Q)
                        a = enables[np.argmax(Q_flatted[enables])]

                    next_s, r, done, _ = self.env.step((a, player))

                    enables = self.env.possible_actions
                    # get next_Q and find next_max_Q
                    if done:
                        max_next_Q = 0
                    else:
                        next_Q = sess.run(self.model.Q, \
                                          feed_dict={self.model.input_s: np.reshape(next_s, (1, self.model.input_length))})
                        next_Q_flatted = np.ravel(next_Q)
                        max_next_Q = np.max(next_Q_flatted[enables])

                    Q_target = Q
                    if i == play_game_times - 1:
                        print('Q', Q[0][a])
                    Q_target[0][a] = r + gamma * max_next_Q
                    if i == play_game_times - 1:
                        print('tQ ', Q_target[0][a])

                    # update
                    _ = sess.run(self.model.update, \
                            feed_dict={self.model.Q_target: Q_target, self.model.input_s: np.reshape(s, (1, self.model.input_length))})
                    s = next_s

                    # change player
                    player ^= 1
                    if done:
                        print('game {} : {}'.format(i, step))
                        if eps > eps_min:
                            eps *= eps_decay
                        break

            # save model
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(self.model_dir, 'parameter.ckpt'))

if __name__ == '__main__':
    agent = RL_QG_agent()
