import copy
from collections import deque

import numpy as np
import tensorflow as tf
import os


class DQN():
    def __init__(self, num_states, num_actions, config, weight_path, log_path, is_training, namespace, variable_scope):
        self.is_training = is_training
        self.weight_path = weight_path
        self.log_path = log_path
        self.config = config
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_size = config.hidden_size_dense
        self.scope = namespace
        self.batch_size = config.batch_size  # How many experiences to use for each training step.
        self.W = []
        self.b = []
        self.init_network()
        self.replay_buffer = DQNExperienceBuffer(config.buffer_size)

        # self.batch_size = 20  # How many experiences to use for each training step.
        self.gamma = config.gamma  # Discount factor on the target Q-values
        self.training_iteration = 0

        self.summary_writer = None
        # save_var = {"W_out": self.W_out, "b_out": self.b_out}
        # summeries = []
        # for i in range(len(self.W)):
        #     save_var["W_" + str(i)] = self.W[i]
        #     save_var["b_" + str(i)] = self.b[i]
            # with tf.name_scope(self.scope + "_weight_summaries"):
            #     summeries.append(tf.summary.histogram(namespace + "_weights" + str(i), self.W[i]))
            # with tf.name_scope(self.scope + "_biases_summaries"):
            #     summeries.append(tf.summary.histogram(namespace + "_biases" + str(i), self.b[i]))

        # for i in range(0, len(self.layers)):
        #     with tf.name_scope(self.scope + "_activation_summaries"):
        #         summeries.append(tf.summary.histogram(namespace + "_activation" + str(i), self.layers[i]))

        # with tf.name_scope(self.scope + "_summaries"):
        #     summeries.append(tf.summary.histogram(namespace + "_weights_out", self.W_out))
        #     summeries.append(tf.summary.histogram(namespace + "_biases_out", self.b_out))
        # summeries.extend(self.variable_summaries(self.td_error, self.scope + "_td_error"))
        # summeries.append(tf.summary.scalar(self.scope + "_loss", self.loss))
        # self.variable_summaries(self.loss, self.scope +"_loss")
        # self.sum_error_merge = tf.summary.merge(summeries)
        save_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope.name)
        self.saver = tf.train.Saver(
            save_var, max_to_keep=1)

    def load_weight(self, sess):
        print('Loading Model...')
        print self.weight_path
        ckpt = tf.train.get_checkpoint_state(self.weight_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save_weights(self, sess, prefix):
        self.saver.save(sess, self.weight_path + '/model-' + str(prefix) + '.cptk')

    def variable_summaries(self, var, name):
        sum_list = []
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries' + name):
            mean = tf.reduce_mean(var)
            sum_list.append(tf.summary.scalar('mean', mean))
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            sum_list.append(tf.summary.scalar('stddev', stddev))
            sum_list.append(tf.summary.scalar('max', tf.reduce_max(var)))
            sum_list.append(tf.summary.scalar('min', tf.reduce_min(var)))
            sum_list.append(tf.summary.histogram('histogram', var))
        return sum_list

    def create_layer(self, prev_layer, W, b, use_tanh):
        if use_tanh:
            if b is None:
                layer = tf.nn.tanh(tf.matmul(prev_layer, W))
            else:
                layer = tf.nn.tanh(tf.add(tf.matmul(prev_layer, W), b))
        else:
            if b is None:
                layer = tf.nn.relu(tf.matmul(prev_layer, W))
            else:
                layer = tf.nn.relu(tf.add(tf.matmul(prev_layer, W), b))
        # layer = tf.contrib.layers.batch_norm(layer,
        #                              center=True, scale=True,
        #                              is_training=self.is_training)
        if self.is_training and self.config.keep_prob < 1:
            layer = tf.nn.dropout(layer, self.config.keep_prob)

        return layer

    def init_network(self):
        summeries = []
        h_size = copy.copy(self.config.hidden_size_dense)
        h_size.insert(0, self.num_states)
        self.reward_avg = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.velocity_avg = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.observations = tf.placeholder(tf.float32, [None, self.num_states],
                                           name="observations")

        self.W_out = tf.get_variable(self.scope + "_W_out", [h_size[-1], self.num_actions], dtype=tf.float32,
                                     trainable=self.is_training)
        self.b_out = tf.get_variable(self.scope + "_b_out", [1, self.num_actions], dtype=tf.float32,
                                     initializer=tf.constant_initializer(self.config.init_constant),
                                     trainable=self.is_training)
        # with tf.name_scope(self.scope + "_summaries"):
        #     summeries.append(tf.summary.histogram(self.scope + "_weights_out", self.W_out))
        #     summeries.append(tf.summary.histogram(self.scope + "_biases_out", self.b_out))
        if self.is_training and self.config.keep_prob < 1:
            self.observations = tf.nn.dropout(self.observations, self.config.keep_prob)

        for i in range(1, len(h_size)):
            self.W.append(tf.get_variable(self.scope + "_W_" + str(i), [h_size[i - 1], h_size[i]], dtype=tf.float32,
                                          trainable=self.is_training))
            self.b.append(tf.get_variable(self.scope + "_b_" + str(i), [1, h_size[i]], dtype=tf.float32,
                                          initializer=tf.constant_initializer(self.config.init_constant),
                                          trainable=self.is_training))
            # with tf.name_scope(self.scope + "_weight_summaries"):
            #     summeries.append(tf.summary.histogram(self.scope + "_weights" + str(i), self.W[-1]))
            # with tf.name_scope(self.scope + "_biases_summaries"):
            #     summeries.append(tf.summary.histogram(self.scope + "_biases" + str(i), self.b[-1]))

        self.layers = [self.create_layer(self.observations, self.W[0], self.b[0], self.config.use_tanh_dense[0])]
        for i in range(1, len(self.W)):
            self.layers.append(self.create_layer(self.layers[i - 1], self.W[i], self.b[i], self.config.use_tanh_dense[i]))

        self.Qout = tf.matmul(self.layers[-1], self.W_out) + self.b_out
        self.probability = tf.nn.sigmoid(self.Qout)
        self.predict = tf.argmax(self.Qout, 1)


        # self.max_Q = tf.max(self.Qout, 1)
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        if not self.is_training:
            return
        # self.td_clamp1 = tf.minimum(self.td_error, self.config.clamp)
        # self.td_clamp2 = tf.maximum(self.td_clamp1, -self.config.clamp)
        self.loss = tf.reduce_mean(self.td_error)
        # self.trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self._lr = tf.Variable(0.0, trainable=False)

        # self.trainer = tf.train.AdamOptimizer(learning_rate=self._lr)
        # self.trainer = tf.train.RMSPropOptimizer(learning_rate=self._lr)
        self.trainer = self.gen_trainer(self.config.optimizer, self._lr)
        self.updateModel = self.trainer.minimize(self.loss)

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        self.reward_sum_merge = tf.summary.merge(self.variable_summaries(self.reward_avg, self.scope + "_reward"))
        self.velocity_sum_merge = tf.summary.merge(self.variable_summaries(self.velocity_avg, self.scope + "_velocity"))
        # self.variable_summaries(self.loss, self.scope +"_loss")

        # for i in range(0, len(self.layers)):
        #     with tf.name_scope(self.scope + "_activation_summaries"):
        #         summeries.append(tf.summary.histogram(self.scope + "_activation" + str(i), self.layers[i]))


        summeries.extend(self.variable_summaries(self.td_error, self.scope + "_td_error"))
        summeries.append(tf.summary.scalar(self.scope + "_loss", self.loss))
        # self.variable_summaries(self.loss, self.scope +"_loss")
        self.sum_error_merge = tf.summary.merge(summeries)


    def reset_episode_buffer(self):
        self.episode_buffer = []
        # pass

    def gen_trainer(self, txt, learn_variable):
        result = None
        if txt in "GradientDescentOptimizer":
            result = tf.train.GradientDescentOptimizer(learning_rate=learn_variable)
        elif txt in "AdamOptimizer":
            result = tf.train.AdamOptimizer(learning_rate=learn_variable)
        elif txt in "RMSPropOptimizer":
            result = tf.train.RMSPropOptimizer(learning_rate=learn_variable)
        return result

    def init_logger(self, sess):
        self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)

    def train_network(self, sess, learning_rate, log_data=False):
        if len(self.replay_buffer.buffer) < 1:
            return
        sess.run(self._lr_update, feed_dict={self._new_lr: learning_rate})

        trainBatch = self.replay_buffer.sample(self.batch_size)  # Get a random batch of experiences.
        # Below we perform the Double-DQN update to the target Q-values
        Q1 = sess.run(self.predict, feed_dict={self.observations: np.vstack(trainBatch[:, 3])})
        Q2 = sess.run(self.Qout, feed_dict={self.observations: np.vstack(trainBatch[:, 3])})
        end_multiplier = -(trainBatch[:, 4] - 1)
        # Q_1 = np.max(Q2[range(self.batch_size), :], axis=1)
        Q_predict = sess.run(self.predict, feed_dict={self.observations: np.vstack(trainBatch[:, 0])})
        Q_current = sess.run(self.Qout, feed_dict={self.observations: np.vstack(trainBatch[:, 0])})
        Q_out_current = Q_current[range(self.batch_size), Q_predict]

        if not self.config.use_double_q:
            Q_max = np.max(Q2, axis=1)
            delta = trainBatch[:, 2] + (self.gamma * Q_max * end_multiplier) - Q_out_current
        else:
            doubleQ = Q2[range(self.batch_size), Q1]
            delta = trainBatch[:, 2] + (self.gamma * doubleQ * end_multiplier) - Q_out_current

        factor = [1.0] * len(delta)
        if self.config.use_QDQN:
            for i in range(len(delta)):
                if delta[i] >= 0:
                    factor[i] = self.config.alpha
                else:
                    factor[i] = self.config.beta

        targetQ = Q_out_current + factor * delta

        # Update the network with our target values.
        if log_data:
            Q, td_error, summary, _ = sess.run([self.Q, self.td_error, self.sum_error_merge, self.updateModel], \
                                  feed_dict={self.observations: np.vstack(trainBatch[:, 0]),
                                             self.targetQ: targetQ, \
                                             self.actions: trainBatch[:, 1]})
            self.summary_writer.add_summary(summary, self.training_iteration)
            self.training_iteration += 1
            # print "target: {0} Q: {1} td_error: {2}".format(targetQ[0], Q[0], td_error[0])
            # print self.batch_size
        else:
            _ = sess.run([self.updateModel], \
                         feed_dict={self.observations: np.vstack(trainBatch[:, 0]),
                                    self.targetQ: targetQ, \
                                    self.actions: trainBatch[:, 1]})

    def store_episode(self, s, a, r, s1, d):
        self.replay_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

    def train_single(self, sess, learning_rate, s, a, r, s1, d):
        # sess.run(self._lr_update, feed_dict={self._new_lr: learning_rate})

        trainBatch = np.array(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
        trainBatch = np.reshape(trainBatch, [1, 5])
        # Below we perform the Double-DQN update to the target Q-values
        predict = sess.run(self.predict, feed_dict={self.observations: np.vstack(trainBatch[:, 3])})
        q_out = sess.run(self.Qout, feed_dict={self.observations: np.vstack(trainBatch[:, 3])})
        end_multiplier = -(trainBatch[:, 4] - 1)

        Q_predict = sess.run(self.predict, feed_dict={self.observations: np.vstack(trainBatch[:, 0])})
        Q_current = sess.run(self.Qout, feed_dict={self.observations: np.vstack(trainBatch[:, 0])})
        Q_out_current = Q_current[range(self.batch_size), Q_predict]
        if not self.config.use_double_q:
            Q_max = np.max(q_out, axis=1)
            delta = trainBatch[:, 2] + (self.gamma * Q_max * end_multiplier) - Q_out_current
        else:
            doubleQ = q_out[range(self.batch_size), predict]
            delta = trainBatch[:, 2] + (self.gamma * doubleQ * end_multiplier) - Q_out_current

        factor = [1.0] * len(delta)
        if self.config.use_QDQN:
            for i in range(len(delta)):
                if delta[i] >= 0:
                    factor[i] = self.config.alpha
                else:
                    factor[i] = self.config.beta

        targetQ = Q_out_current + factor * delta
        # Update the network with our target values.
        td_error = sess.run([self.td_error], \
                            feed_dict={self.observations: np.vstack(trainBatch[:, 0]),
                                       self.targetQ: targetQ, \
                                       self.actions: trainBatch[:, 1]})
        # print q_1, q_out

        return predict, q_out, td_error


    def store_rewards(self, sess, rewards, counter):
        summary_str = sess.run(self.reward_sum_merge, \
                               feed_dict={self.reward_avg: np.vstack(rewards)})
        self.summary_writer.add_summary(summary_str, counter)

    def log_velocity(self, sess, velocity, counter):
        summary_str = sess.run(self.velocity_sum_merge, \
                               feed_dict={self.velocity_avg: np.vstack(velocity)})
        self.summary_writer.add_summary(summary_str, counter)

    def predict_action(self, sess, current_state):
        a = sess.run([self.predict], \
                     feed_dict={self.observations: [current_state]})
        return a

    @staticmethod
    def generate_path(src, cfg):
        output = src
        output += "lr" + str(cfg.max_learning_rate) + "_"
        output += "rnn_"
        for h in range(len(cfg.hidden_size_rnn)):
            if cfg.use_tanh_rnn[h]:
                output += "tanh" + str(cfg.hidden_size_rnn[h]) + "_"
            else:
                output += "relu" + str(cfg.hidden_size_rnn[h]) + "_"
        output += "dense_"
        for h in range(len(cfg.hidden_size_dense)):
            if cfg.use_tanh_dense[h]:
                output += "tanh" + str(cfg.hidden_size_dense[h]) + "_"
            else:
                output += "relu" + str(cfg.hidden_size_dense[h]) + "_"
        output += "_batch" + str(cfg.batch_size)
        output += "_trace_length" + str(cfg.trace_length) + "_"
        output += "forget_bias" + str(cfg.forget_bias) + "_"
        output += "init_value" + str(cfg.init_scale) + "_"
        output += "keep_prob" + str(cfg.keep_prob) + "_"
        try:
            if cfg.use_double_q:
                output += "double_q_on_"
            else:
                output += "double_q_off_"
            if cfg.use_QDQN:
                output += "QDQN_on_"
            else:
                output += "QDQN_off_"
            output += "alpha" + str(cfg.alpha) + "_"
            output += "beta" + str(cfg.beta) + "_"
        except AttributeError:
            print "doesn't have use_double_q or use_QDQN"
        output += cfg.optimizer
        print output
        return output

    @staticmethod
    def translate(value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)

    @staticmethod
    def create_nn_folders(cfg, weight_path, log_path):
        load = True
        gen_weight_path = cfg.weight_folder
        gen_log_path = cfg.log_folder

        if cfg.weight_folder is None:
            load = False
            gen_weight_path = DQN.generate_path(weight_path, cfg)
            i = 0
            while os.path.exists(gen_weight_path + "_" + str(i)):
                i += 1
            gen_weight_path += "_" + str(i)

        if cfg.log_folder is None:
            gen_log_path = DQN.generate_path(log_path, cfg)
            i = 0
            while os.path.exists(gen_log_path + "_" + str(i)):
                i += 1
            gen_log_path += "_" + str(i)

        # Make a path for our model to be saved in.
        if not os.path.exists(gen_log_path):
            os.makedirs(gen_log_path)
        if not os.path.exists(gen_weight_path):
            os.makedirs(gen_weight_path)

        cfg.weight_folder = gen_weight_path
        cfg.log_folder = gen_log_path

        return gen_weight_path, gen_log_path, load


class DQNExperienceBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        # sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        sample_count = 0
        while sample_count < batch_size:
            point = np.random.randint(0, len(self.buffer))

            # data1 = [self.buffer_x[i] for i in range(point, point + trace_length)]
            # data2 = [self.buffer_y[i] for i in range(point+1, point + trace_length + 1)]
            sampledTraces.append([self.buffer[point]])
            sample_count += 1
        sampledTraces = np.array(sampledTraces)
        result = np.reshape(sampledTraces, [batch_size, 5])
        return result
