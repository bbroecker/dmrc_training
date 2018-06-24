import inspect
from collections import deque

import numpy as np
import tensorflow as tf


def leaky_relu(x, alpha=0.1):
    return tf.maximum(x*alpha, x)

class DQN_RNN():
    def __init__(self, num_states, num_actions, num_drones, config, weight_path, log_path, is_training, namespace, variable_scope):
        self.is_training = is_training
        self.weight_path = weight_path
        self.log_path = log_path
        self.num_drones = num_drones
        self.config = config
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_size = config.hidden_size_rnn[0]
        self.scope = namespace
        self.batch_size = config.batch_size  # How many experiences to use for each training step.
        self.trace_length = config.trace_length  # How long each experience trace will be when training
        self.init_network()
        self.replay_buffer = ExperienceBuffer(config.buffer_size, num_drones)

        # self.batch_size = 20  # How many experiences to use for each training step.
        self.gamma = config.gamma  # Discount factor on the target Q-values
        self.training_iteration = 0

        self.summary_writer = None
        save_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope.name)

        self.saver = tf.train.Saver(
            save_var, max_to_keep=1)

    def load_weight(self, sess):
        print 'Loading Model... {0}'.format(self.weight_path)
        ckpt = tf.train.get_checkpoint_state(self.weight_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save_weights(self, sess, prefix):
        self.saver.save(sess, self.weight_path + '/model-' + str(prefix) + '.cptk')

    def save_weights_to_folder(self, sess, prefix, path):
        self.saver.save(sess, path + '/model-' + str(prefix) + '.cptk')


    def variable_summaries(self, var, name):
        tmp = []
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries' + name):
            mean = tf.reduce_mean(var)
            tmp.append(tf.summary.scalar('mean', mean))
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tmp.append(tf.summary.scalar('stddev', stddev))
            tmp.append(tf.summary.scalar('max', tf.reduce_max(var)))
            tmp.append(tf.summary.scalar('min', tf.reduce_min(var)))
            tmp.append(tf.summary.histogram('histogram', var))

        return tmp



    def lstm_cell(self):
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is unfortunately not
        # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
        # an argument check here:
        # acti = tf.tanh if self.config.use_tanh_rnn[0] else tf.nn.relu
        # acti = tf.tanh if self.config.use_tanh_rnn[0] else leaky_relu
        acti = tf.tanh if self.config.use_tanh_rnn[0] else tf.nn.relu
        # initializer = tf.random_normal_initializer(stddev=self.config.init_scale)

        if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.LSTMCell.__init__).args:
            print "reuse LSTMCell {0}".format(self.config.hidden_size_rnn[0])
            return tf.contrib.rnn.LSTMCell(
                self.config.hidden_size_rnn[0], forget_bias=self.config.forget_bias, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size,
            #     reuse=tf.get_variable_scope().reuse, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size, reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.LSTMCell(
                self.config.hidden_size_rnn[0], forget_bias=self.config.forget_bias, state_is_tuple=True, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size)

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        # print output.get_shape(), batch_size, max_length
        out_size = int(output.get_shape()[1])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def init_network(self):
        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.reward_avg = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.goal_per_sec = tf.placeholder(dtype=tf.float32)
        self.run_percentage = tf.placeholder(dtype=tf.float32)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processin
        # and then returned to [batch x units] when sent through the upper levles.
        self.tf_batch_size = tf.placeholder(dtype=tf.int32)
        self.observations = tf.placeholder(tf.float32, [None, self.num_states], name="observations")
        if self.is_training and self.config.keep_prob < 1:
            self.observations = tf.nn.dropout(self.observations, self.config.keep_prob)

        self.observations_rrn = tf.reshape(self.observations,
                                           [self.tf_batch_size, self.trainLength, self.num_states])
        attn_cell = self.lstm_cell
        if self.is_training and self.config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(self.lstm_cell(), output_keep_prob=self.config.keep_prob)

        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(len(self.config.hidden_size_rnn))],
                                                        state_is_tuple=True)
        self.state_in = self.stacked_lstm.zero_state(self.tf_batch_size, tf.float32)

        self.rnn, self.rnn_state = tf.nn.dynamic_rnn( \
            inputs=self.observations_rrn, cell=self.stacked_lstm, dtype=tf.float32, initial_state=self.state_in,
            scope=self.scope + '_rnn', sequence_length=self.trainLength)

        self.rnn = tf.reshape(self.rnn, shape=[-1, self.config.hidden_size_rnn[0]])

        # self.W_1 = tf.get_variable(self.scope + "_W_1", [self.config.hidden_size, self.config.hidden_size],
        #                              dtype=tf.float32)
        # self.b_1 = tf.get_variable(self.scope + "_b_1", [1, self.config.hidden_size], dtype=tf.float32)
        # self.layer = tf.nn.relu(tf.matmul(self.rnn, self.W_1) + self.b_1)

        # self.streamA,self.streamV = tf.split(self.rnn,2,1)
        # self.AW = tf.Variable(tf.random_normal([self.config.hidden_size//2, 5]))
        # self.VW = tf.Variable(tf.random_normal([self.config.hidden_size//2, 1]))
        # self.Advantage = tf.matmul(self.streamA,self.AW)
        # self.Value = tf.matmul(self.streamV,self.VW)
        # self.salience = tf.gradients(self.Advantage, self.observations)
        # self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        # self.predict = tf.argmax(self.Qout, 1)

        self.W_out = tf.get_variable(self.scope + "_W_out", [self.config.hidden_size_rnn[0], self.num_actions],
                                     dtype=tf.float32)
        self.b_out = tf.get_variable(self.scope + "_b_out", [1, self.num_actions], dtype=tf.float32)
        # self.Qout = tf.nn.tanh(tf.add(tf.matmul(self.rnn, self.W1), self.b1))
        self.Qout = tf.matmul(self.rnn, self.W_out) + self.b_out

        # self.probability = tf.nn.sigmoid(self.Qout)
        self.predict = tf.argmax(self.Qout, 1)

        if not self.is_training:
            return
        # self.max_Q = tf.max(self.Qout, 1)
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        # self.td_clamp1 = tf.minimum(self.td_error, self.config.clip)
        # self.td_clamp2 = tf.maximum(self.td_clamp1, -self.config.clip)
        # self.loss = tf.reduce_mean(self.td_error)


        maskA = tf.zeros([self.config.batch_size, self.config.min_history])
        maskB = tf.ones([self.config.batch_size, self.config.update_length])
        maskC = tf.zeros(
            [self.config.batch_size, self.config.trace_length - self.config.update_length - self.config.min_history])
        mask = tf.concat([maskA, maskB, maskC], 1)
        mask = tf.reshape(mask, [-1])
        self.td_error = tf.boolean_mask(self.td_error, tf.cast(mask, tf.bool))
        self.loss = tf.reduce_mean(self.td_error, name=self.scope + "_loss")

        #
        #
        # self.maskA = tf.zeros([self.tf_batch_size, self.trainLength // 2])
        # self.maskB = tf.ones([self.tf_batch_size, self.trainLength // 2])
        # self.mask = tf.concat([self.maskA, self.maskB], 1)
        # self.mask = tf.reshape(self.mask, [-1])
        # self.loss = tf.reduce_mean(self.td_error * self.mask)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                          self.config.max_grad_norm)

        # self.loss = tf.reduce_mean(self.td_error)
        # self.trainer = tf.train.AdamOptimizer(learning_rate=0.01)
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self._lr = tf.Variable(0.0, trainable=False)
        # self.trainer = tf.train.AdamOptimizer(learning_rate=self._lr)
        self.trainer = self.gen_trainer(self.config.optimizer, self._lr)
        self.updateModel = self.trainer.apply_gradients(zip(grads, tvars))
        # self.updateModel = self.trainer.minimize(self.loss)

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

        # self.variable_summaries(self.td_error, "_td_error")
        self.cost_sum = tf.summary.scalar(self.scope + "_loss", self.td_error)
        # self.td_sum = tf.summary.scalar(self.scope + "_td_error", tf.reduce_mean(self.td_error))
        # self.variable_summaries(self.td_clamp2, "_td_clamp")
        # self.variable_summaries(self.loss, "_loss")
        # self.variable_summaries(self.Q, "Q")
        tmp_sums = []
        tmp_sums.extend(self.variable_summaries(self.td_error, self.scope + "_td_error"))
        tmp_sums.append(tf.summary.scalar(self.scope + "_loss", self.loss))
        self.reward_sum_merge = tf.summary.merge(self.variable_summaries(self.reward_avg, self.scope + "_reward"))
        self.goal_per_sec_sum = tf.summary.scalar(self.scope + "_goal_per_sec", self.goal_per_sec)
        self.run_percentage_sum = tf.summary.scalar(self.scope + "_run_completed", self.run_percentage)
        # self.variable_summaries(self.loss, self.scope +"_loss")
        self.sum_error_merge = tf.summary.merge(tmp_sums)

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

    def store_rewards(self, sess, rewards, counter):
        summary_str = sess.run(self.reward_sum_merge, \
                               feed_dict={self.reward_avg: np.vstack(rewards)})
        self.summary_writer.add_summary(summary_str, counter)

    def store_goal_per_sec(self, sess, goal_per_sec, counter):
        summary_str = sess.run(self.goal_per_sec_sum, \
                               feed_dict={self.goal_per_sec: goal_per_sec})
        self.summary_writer.add_summary(summary_str, counter)

    def store_completion_percentage(self, sess, competion, counter):
        summary_str = sess.run(self.run_percentage_sum, \
                               feed_dict={self.run_percentage: competion})
        self.summary_writer.add_summary(summary_str, counter)

    def gen_init_state(self, sess, batch_size):
        return sess.run([self.state_in], feed_dict={self.tf_batch_size: batch_size})

    def init_logger(self, sess):
        self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)

    def train_network(self, sess, learning_rate, log_data=False):
        if len(self.replay_buffer.buffer) < 1:
            return

        sess.run(self._lr_update, feed_dict={self._new_lr: learning_rate})


        trainBatch = self.replay_buffer.sample(self.batch_size, self.trace_length)  # Get a random batch of experiences.
        # Below we perform the Double-DQN update to the target Q-values
        for i in range(self.config.max_epoch):
            state_train = sess.run([self.state_in], feed_dict={self.tf_batch_size: self.config.batch_size})
            Q1 = sess.run(self.predict, feed_dict={ \
                self.observations: np.vstack(trainBatch[:, 3]), \
                self.trainLength: self.trace_length, self.state_in: state_train, self.tf_batch_size: self.batch_size})
            Q2 = sess.run(self.Qout, feed_dict={ \
                self.observations: np.vstack(trainBatch[:, 3]), \
                self.trainLength: self.trace_length, self.state_in: state_train,
                self.tf_batch_size: self.batch_size})

            end_multiplier = -(trainBatch[:, 4] - 1)
            doubleQ = Q2[range(self.batch_size * self.trace_length), Q1]
            targetQ = trainBatch[:, 2] + (self.gamma * doubleQ * end_multiplier)
            # Update the network with our target values.
        # print "obs {0}\n obs_rnn{1}".format(trainBatch[:,3], obs_rnn)
            if log_data and i == self.config.max_epoch - 1:
                summary, _ = sess.run([self.sum_error_merge, self.updateModel], \
                                      feed_dict={self.observations: np.vstack(trainBatch[:, 0]),
                                                 self.targetQ: targetQ, \
                                                 self.actions: trainBatch[:, 1], self.trainLength: self.trace_length, \
                                                 self.state_in: state_train, self.tf_batch_size: self.batch_size})
                self.summary_writer.add_summary(summary, self.training_iteration)
                self.training_iteration += 1
            else:
                _ = sess.run(self.updateModel, \
                                      feed_dict={self.observations: np.vstack(trainBatch[:, 0]),
                                                 self.targetQ: targetQ, \
                                                 self.actions: trainBatch[:, 1], self.trainLength: self.trace_length, \
                                                 self.state_in: state_train, self.tf_batch_size: self.batch_size})

    def single_train_network(self, sess, learning_rate, rrn_state, s, a, r, s1, d):

        sess.run(self._lr_update, feed_dict={self._new_lr: learning_rate})
        trainBatch = np.array(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
        trainBatch = np.reshape(trainBatch, [1, 5])

        # state_train = sess.run([self.state_in], feed_dict={self.tf_batch_size: self.config.batch_size})
        # trainBatch = self.replay_buffer.sample(self.batch_size, self.trace_length)  # Get a random batch of experiences.
        # Below we perform the Double-DQN update to the target Q-values
        predict = sess.run(self.predict, feed_dict={ \
            self.observations: np.vstack(trainBatch[:, 3]), \
            self.trainLength: 1, self.state_in: rrn_state, self.tf_batch_size: 1})
        q_out = sess.run(self.Qout, feed_dict={ \
            self.observations: np.vstack(trainBatch[:, 3]), \
            self.trainLength: 1, self.state_in: rrn_state,
            self.tf_batch_size: 1})

        end_multiplier = -(trainBatch[:, 4] - 1)
        doubleQ = q_out[range(1 * 1), predict]
        targetQ = trainBatch[:, 2] + (self.gamma * doubleQ * end_multiplier)
        # Update the network with our target values.
        td_error = sess.run(self.td_error, \
                                 feed_dict={self.observations: np.vstack(trainBatch[:, 0]),
                                            self.targetQ: targetQ, \
                                            self.actions: trainBatch[:, 1], self.trainLength: 1, \
                                            self.state_in: rrn_state, self.tf_batch_size: 1})
        # print "<--------"
        # print q_out
        # print "---------"
        # print rnn
        # print "-------->"
        return predict, q_out, td_error

    def predict_action(self, sess, old_rr_state, s):
        a, state1 = sess.run([self.predict, self.rnn_state], \
                             feed_dict={self.observations: [s], self.trainLength: 1,
                                        self.state_in: old_rr_state, self.tf_batch_size: 1})
        return a, state1

    def store_episode(self, s, a, r, s1, d, drone_id, episode_id):
        self.replay_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]), drone_id, episode_id)

    def new_rrn_state(self, sess, old_rr_state, s):
        return sess.run(self.rnn_state, \
                        feed_dict={self.observations: [s], self.trainLength: 1,
                                   self.state_in: old_rr_state, self.tf_batch_size: 1})


# class ExperienceBuffer:
#     def __init__(self, buffer_size, num_drones):
#         self.buffer = []
#         self.buffer_size = buffer_size
#
#     def add(self, experience):
#
#         if len(self.buffer) + 1 >= self.buffer_size:
#             self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
#         self.buffer.append(experience)
#
#     def sample(self, batch_size, trace_length):
#         # sampled_episodes = random.sample(self.buffer, batch_size)
#         sampledTraces = []
#         sample_count = 0
#         while sample_count < batch_size:
#             random_episode = random.sample(self.buffer, 1)[0]
#             # print random_episode
#             point = np.random.randint(0, len(random_episode) + 1 - trace_length)
#             sampledTraces.append(random_episode[point:point + trace_length])
#             sample_count += 1
#         sampledTraces = np.array(sampledTraces)
#         return np.reshape(sampledTraces, [batch_size * trace_length, 5])
#         # return np.reshape(np.array(random.sample(self.buffer, batch_size)), [batch_size, 5])



class ExperienceBuffer:
    def __init__(self, buffer_size, num_buckets):
        self.buffer = []
        self.current_episode = []
        for i in range(num_buckets):
            self.buffer.append(deque(maxlen=buffer_size))
            self.current_episode.append(-1)
        self.buffer_size = buffer_size
        self.num_buckets = num_buckets

    def add(self, sample, buckets_id, episode_id):

        if episode_id != self.current_episode[buckets_id]:
            self.buffer[buckets_id].append([])
            self.current_episode[buckets_id] = episode_id
        self.buffer[buckets_id][-1].append(sample)

    def sample(self, batch_size, trace_length):
        # sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        sample_count = 0
        while sample_count < batch_size:
            bucket = np.random.randint(0, self.num_buckets)
            episode = np.random.randint(0, len(self.buffer[bucket]))
            if len(self.buffer[bucket][episode]) - trace_length < 1:
                continue
            point = np.random.randint(0, len(self.buffer[bucket][episode]) - trace_length + 1)
            tmp = []
            for i in range(point, point + trace_length):
                tmp.append(self.buffer[bucket][episode][i])
            # data1 = [self.buffer_x[i] for i in range(point, point + trace_length)]
            # data2 = [self.buffer_y[i] for i in range(point+1, point + trace_length + 1)]
            sampledTraces.append(tmp)
            sample_count += 1
        sampledTraces = np.array(sampledTraces)
        result = np.reshape(sampledTraces, [batch_size * trace_length, 5])
        return result
