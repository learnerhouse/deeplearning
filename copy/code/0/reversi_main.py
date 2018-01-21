import tensorflow as tf
import os
import gym
import numpy as np

class Simple_model:
    '''
        a simple neural network
        use flatten input
    '''
    def __init__(self, env):
        self.input_length = env.board_size ** 2 * 3
        self.input_s = tf.placeholder(shape=[1, self.input_length], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform(shape=[self.input_length, env.action_space.n], minval=0, maxval=0.001))
        self.Q = tf.matmul(self.input_s, self.W)
        self.predict_action = tf.argmax(self.Q, 1)

        self.Q_target = tf.placeholder(shape=[1, env.action_space.n], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_target - self.Q))
        self.update = tf.train.GradientDescentOptimizer(1e-4).minimize(self.loss)
        self.init = tf.global_variables_initializer()

class RL_QG_agent:
    def __init__(self):
        # loading_model = False
        loading_model = True
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Reversi_model")
        self.env = gym.make('Reversi8x8-v0')
        # self.init_model()
        self.model = Simple_model(env = self.env)
        if not loading_model:
            self.train_simple_model()

    def init_model(self):
        '''
        # 定义自己的 网络
        self.sess = tf.Session()
        # 补全代码
        '''
        # self.sess = tf.Session()
        # self.saver = tf.train.Saver()

    def place(self,state,enables,player):
        # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
        # action 表示的是 要下的位置。
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(self.model_dir, 'parameter.ckpt'))
            Q = sess.run(self.model.Q, \
                                feed_dict={self.model.not input_s: np.reshape(state, (1, self.model.input_length))})
            Q = np.ravel(Q)
            action = enables[np.argmax(Q[enables])]
        return action

    def save_model(self):  # 保存 模型
        self.saver.save(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))

    def load_model(self):# 重新导入模型
        self.saver.restore(self.sess, os.path.join(self.model_dir, 'parameter.ckpt'))


    def train_simple_model(self):
        # Start training
        eps = 1
        gamma = 0.99
        play_game_times = 10000
        eps_min = 0.01
        eps_decay = 0.99999

        with tf.Session() as sess:
            sess.run(self.model.init)
            for i in range(play_game_times):
                print('game {}'.format(i))
                s = self.env.reset()
                done = False
                step = 0
                player = 0
                while True:
                    step += 1
                    # get Q
                    a, Q = sess.run([self.model.predict_action, self.model.Q],\
                                    feed_dict={self.model.input_s: np.reshape(s, (1, self.model.input_length))})
                    if np.random.rand(1) < eps:
                        # a = self.env.action_space.sample()
                        a = np.random.choice(self.env.possible_actions)
                    next_s, r, done, _ = self.env.step((a, player))

                    # get next_Q and find next_max_Q
                    if done:
                        max_next_Q = 0
                    else:
                        next_Q = sess.run(self.model.Q, \
                                          feed_dict={self.model.input_s: np.reshape(next_s, (1, self.model.input_length))})
                        max_next_Q = np.max(next_Q)
                    # print('nQ', next_Q)
                    Q_target = Q
                    # print('Q', Q[0][a])
                    Q_target[0][a] = r + gamma * max_next_Q

                    # update
                    _, W = sess.run([self.model.update, self.model.W], \
                            feed_dict={self.model.Q_target: Q_target, self.model.input_s: np.reshape(s, (1, self.model.input_length))})
                    s = next_s

                    # change player
                    player ^= 1
                    if done:
                        print('step = ', step)
                        if eps > eps_min:
                            eps *= eps_decay
                        break

            # save model
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(self.model_dir, 'parameter.ckpt'))

if __name__ == '__main__':
    agent = RL_QG_agent()
