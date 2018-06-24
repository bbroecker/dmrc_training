import inspect
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def np_mod(x,y):
    return (x % y).astype(np.float32)

def modgrad(op, grad):
    x = op.inputs[0] # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
    y = op.inputs[1] # the second argument

    return grad * 1, grad * tf.negative(tf.floordiv(x, y))

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def tf_mod(x,y, name=None):

    with ops.op_scope([x, y], name, "mod") as name:
        z = py_func(np_mod,
                        [x,y],
                        [tf.float32],
                        name=name,
                        grad=modgrad)  # <-- here's the call to the gradient
        return z[0]


class DefaultConfig(object):
    num_layers = 2
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    state_count = 4
    num_steps = 10
    hidden_size = 650
    # max_epoch = 6
    # max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    use_tanh = False
    buffer_size = 50000
    forget_bias = 0.0
    use_fp16 = False
    # vocab_size = 10000



class GeneralRRNDiscreteModel():
    def __init__(self, num_drones, config, is_training, log_path, weight_path, namespace, variable_scope):
        self.config = config
        self.num_drones = num_drones
        self.log_path = log_path
        self.weight_path = weight_path
        self.is_training = is_training
        self.namespace = namespace
        self.W = []
        self.b = []
        self.W_out = None
        self.init_network()
        self.replay_buffer = ExperienceDualBuffer(config, num_drones)
        self.log_count = 0
        self.valid_counter = 0
        summeries = []

        # Execute the LSTM cell here in any way, for example:
        # Retrieve just the LSTM variables.

        with tf.name_scope(self.namespace + "_summaries"):
            summeries.append(tf.summary.histogram(namespace + "_weights_out", self.W_out))
            summeries.append(tf.summary.histogram(namespace + "_biases_out", self.b_out))
        if is_training:
            summeries.append(self.cost_sum)
            summeries.append(self.acc_sum)
        self.sum_merge = tf.summary.merge(summeries)

        save_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope.name)
        self.saver = tf.train.Saver(
            save_var, max_to_keep=1)


    def load_weight(self, sess):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(self.weight_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save_weights(self, sess, prefix):
        self.saver.save(sess, self.weight_path + '/model-' + str(prefix) + '.cptk')

    def save_weights_to_folder(self, sess, prefix, path):
        self.saver.save(sess, path + '/model-' + str(prefix) + '.cptk')

    def data_type(self):
        return tf.float16 if self.config.use_fp16 else tf.float32

    def lstm_cell(self):
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is unfortunately not
        # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
        # an argument check here:
        acti = tf.tanh if self.config.use_tanh_rnn[0] else tf.nn.relu

        if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(
                self.config.hidden_size_rnn[0], forget_bias=self.config.forget_bias, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size, reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                self.config.hidden_size_rnn[0], forget_bias=self.config.forget_bias, state_is_tuple=True, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size)

    def init_network(self):
        self.trace_length = tf.placeholder(dtype=tf.int32)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.tf_batch_size = tf.placeholder(dtype=tf.int32)
        self.observations = tf.placeholder(tf.float32, [None, self.config.state_count], name=self.namespace + "_observations")
        self.W_out = tf.get_variable(self.namespace + "_W_out", [self.config.hidden_size_rnn[0], self.config.output_size], dtype=self.data_type())
        self.b_out = tf.get_variable(self.namespace + "_b_out", [1, self.config.output_size], dtype=self.data_type())
        if self.is_training and self.config.keep_prob < 1:
            self.observations = tf.nn.dropout(self.observations, self.config.keep_prob)

        self.observations_rrn = tf.reshape(self.observations, [self.tf_batch_size, self.trace_length, self.config.state_count])

        attn_cell = self.lstm_cell
        if self.is_training and self.config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(self.lstm_cell(), output_keep_prob=self.config.keep_prob)

        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(len(self.config.hidden_size_rnn))], state_is_tuple=True)
        self.initial_state = self.stacked_lstm.zero_state(self.tf_batch_size, self.data_type())
        rnn_output, self.rnn_state = tf.nn.dynamic_rnn( \
            inputs=self.observations_rrn, cell=self.stacked_lstm, dtype=self.data_type(), initial_state=self.initial_state,
            scope=self.namespace + '_rnn')

        rnn_output = tf.reshape(rnn_output, shape=[-1, self.config.hidden_size_rnn[0]])

        self.output_layer = tf.add(tf.matmul(rnn_output, self.W_out), self.b_out)

        self.predict = tf.argmax(self.output_layer, 1)

        self.Y = tf.placeholder(shape=[None, self.config.output_size], dtype=tf.float32)

        # self.maskA = tf.zeros([self.tf_batch_size, self.trace_length // 2], name=self.namespace + "_Mask_A")
        # self.maskB = tf.ones([self.tf_batch_size, self.trace_length // 2], name=self.namespace + "_Mask_B")
        # self.mask = tf.concat([self.maskA, self.maskB], 1, name=self.namespace + "_Mask")
        # self.mask = tf.reshape(self.mask, [-1])
        # self.loss = tf.reduce_mean(self.td_error * self.mask, name=self.namespace + "_loss")
        # self.loss = tf.reduce_mean(self.td_error * self.mask, name=self.namespace + "_loss")
        with tf.name_scope("accuracy"):
            self.correct_pred1 = tf.equal(tf.argmax(self.Y, 1), self.predict)  # Count correct predictions
            maskA_acc = tf.zeros([self.tf_batch_size, self.config.min_history])
            maskB_acc = tf.ones([self.tf_batch_size, self.config.accuracy_length])
            maskC_acc = tf.zeros(
                [self.tf_batch_size,
                 self.config.trace_length - self.config.accuracy_length - self.config.min_history])
            mask_acc = tf.concat([maskA_acc, maskB_acc, maskC_acc], 1)
            mask_acc = tf.reshape(mask_acc, [-1])
            self.correct_pred2 = tf.boolean_mask(self.correct_pred1, tf.cast(mask_acc, tf.bool))
            self.shape_test = tf.shape(self.correct_pred2)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred2, "float"))  # Cast boolean to float to average
            # Add scalar summary for accuracy tensor
            self.acc_sum = tf.summary.scalar("accuracy", self.accuracy)
            validation = tf.summary.scalar("valid_accuracy", self.accuracy)
            test = []
            test.append(validation)
            self.sum_validation = tf.summary.merge(test)

        if not self.is_training:
            return

        self.error = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.Y, name=None)

        maskA = tf.zeros([self.tf_batch_size, self.config.min_history])
        maskB = tf.ones([self.tf_batch_size, self.config.update_length])
        maskC = tf.zeros(
            [self.tf_batch_size, self.config.trace_length - self.config.update_length - self.config.min_history])
        mask = tf.concat([maskA, maskB, maskC], 1)
        mask = tf.reshape(mask, [-1])
        self.error = tf.boolean_mask(self.error, tf.cast(mask, tf.bool))

        self.loss = tf.reduce_mean(self.error)
        self.cost_sum = tf.summary.scalar("angle_cost_function", self.loss)
        self.acc_sum = tf.summary.scalar("angle_accuracy", self.accuracy)
        self._lr = tf.Variable(0.0, trainable=False)

        # self.trainer = tf.train.RMSPropOptimizer(self._lr, 0.9)
        # self.trainer = tf.train.GradientDescentOptimizer(self._lr)
        # self.trainer = tf.train.AdamOptimizer(self._lr)
        # self.updateModel = optimizer.apply_gradients(
        #     zip(grads, tvars),
        #     global_step=tf.contrib.framework.get_or_create_global_step())
        self.trainer = self.gen_trainer(self.config.optimizer, self._lr)
        self.updateModel = self.trainer.minimize(self.loss)


        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        # self.updateModel = self.trainer.minimize(self.loss)

    def gen_trainer(self, txt, learn_variable):
        result = None
        if txt in "GradientDescentOptimizer":
            result = tf.train.GradientDescentOptimizer(learning_rate=learn_variable)
        elif txt in "AdamOptimizer":
            result = tf.train.AdamOptimizer(learning_rate=learn_variable)
        elif txt in "RMSPropOptimizer":
            result = tf.train.RMSPropOptimizer(learning_rate=learn_variable)
        return result

    def reset_buffer(self):
        self.replay_buffer = ExperienceDualBuffer(self.config, self.num_drones)

    def store_episodes(self):
        self.replay_buffer.add(self.episode_buffer.buffer)

    def gen_init_state(self,sess, batch_size):
        init_state = sess.run([self.initial_state], feed_dict={self.tf_batch_size: batch_size})
        return init_state

    def train_network(self, sess, learn_rate, log = False):
        sess.run(self._lr_update, feed_dict={self._new_lr: learn_rate})
        state_train = sess.run([self.initial_state], feed_dict={self.tf_batch_size: self.config.batch_size})
        trainBatch = self.replay_buffer.sample(self.config.batch_size, self.config.trace_length)  # Get a random batch of experiences.
        x = trainBatch[:, 0]
        y = trainBatch[:, 1]
        if log:
            # print "write stuff x: {0}".format(x)
            # print "write stuff y: {0}".format(y)
            summary_str, _ = sess.run([self.sum_merge, self.updateModel], \
                                      feed_dict={self.observations: np.vstack(x),
                                                 self.Y: np.vstack(y),\
                                                 self.trace_length: self.config.trace_length,\
                                                 self.initial_state: state_train, self.tf_batch_size: self.config.batch_size})
            self.summary_writer.add_summary(summary_str, self.log_count)
            self.log_count += 1
        else:
            _ = sess.run([self.updateModel], \
                         feed_dict={self.observations: np.vstack(x),
                                    self.Y: np.vstack(y), \
                                    self.trace_length: self.config.trace_length, \
                                    self.initial_state: state_train, self.tf_batch_size: self.config.batch_size})

    def init_logger(self, sess):
        self.summary_writer = tf.summary.FileWriter(self.log_path, sess.graph)


    def predict_angle(self, sess, s, old_rr_state):
        a, state1 = sess.run([self.predict, self.rnn_state], \
                             feed_dict={self.observations: [s], self.trace_length: 1,
                                        self.initial_state: old_rr_state, self.tf_batch_size: 1})
        return a, state1


    def validate(self, sess):
        state_train = sess.run([self.initial_state], feed_dict={self.tf_batch_size: self.config.valid_batch_size})
        trainBatch = self.replay_buffer.sample(self.config.valid_batch_size, self.config.trace_length)  # Get a random batch of experiences.
        x = trainBatch[:, 0]
        y = trainBatch[:, 1]

        # print "write stuff x: {0}".format(x)
        # print "write stuff y: {0}".format(y)
        accuracy, summary_str = sess.run([self.accuracy, self.sum_validation], \
                                  feed_dict={self.observations: np.vstack(x),
                                             self.Y: np.vstack(y), self.trace_length: self.config.trace_length, \
                                    self.initial_state: state_train, self.tf_batch_size: self.config.valid_batch_size})
        self.summary_writer.add_summary(summary_str, self.valid_counter)
        self.valid_counter += 1
        return accuracy


    def store_episode(self, state, y, bucket_id, type, episode):
        self.replay_buffer.add(state, y, bucket_id, type, episode)
        # self.replay_buffer.extend(
        #     np.reshape(np.array([state, angle]), [1, 2]))  # Save the experience to our episode buffer.

    def new_rrn_state(self,sess, state, state_rrn):
        return sess.run(self.rnn_state, \
                        feed_dict={self.observations: [state], self.trace_length: 1,
                                   self.initial_state: state_rrn, self.tf_batch_size: 1})

class ExperienceDualBuffer:
    def __init__(self, config, num_bucket):
        self.buffer_x = []
        self.buffer_y = []
        self.current_episode = []
        for t in range(config.bucket_types):
            self.buffer_x.append([])
            self.buffer_y.append([])
            self.current_episode.append([])
            for n in range(num_bucket):
                self.current_episode[-1].append(0)
                self.buffer_x[t].append(deque(maxlen=config.t_bucketsize[t]))
                self.buffer_y[t].append(deque(maxlen=config.t_bucketsize[t]))
        self.config = config
        self.num_bucket = num_bucket

    def add(self, x, y, bucket_id, type, episode):
        if episode != self.current_episode[type][bucket_id] or len(self.buffer_x[type][bucket_id]) == 0:
            self.current_episode[type][bucket_id] = episode
            self.buffer_x[type][bucket_id].append([])
            self.buffer_y[type][bucket_id].append([])
        self.buffer_x[type][bucket_id][-1].append(x)
        self.buffer_y[type][bucket_id][-1].append(y)

    def sample(self, batch_size, trace_length):
        # sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        sample_count = 0
        while sample_count < batch_size:
            redo = False
            try:
                bucket = np.random.randint(0, self.num_bucket)
                type = np.random.randint(0, self.config.bucket_types)
                episode = np.random.randint(0, len(self.buffer_x[type][bucket]))
                if len(self.buffer_x[type][bucket][episode]) - trace_length - 1 < 1:
                    # print "type {} epi len{}".format(type, len(self.buffer_x[type][bucket][episode]))
                    continue
                point = np.random.randint(0, len(self.buffer_x[type][bucket][episode]) - trace_length - 1)
                tmp = []

                for i in range(point, point + trace_length):
                    tmp.append([self.buffer_x[type][bucket][episode][i], self.generate_y(self.buffer_y[type][bucket][episode][i])])
            except IndexError:
                print "caught index error"
                redo = True
            except ValueError:
                print "ValueError"
                redo = True
            if redo:
                continue
            # data1 = [self.buffer_x[i] for i in range(point, point + trace_length)]
            # data2 = [self.buffer_y[i] for i in range(point+1, point + trace_length + 1)]
            sampledTraces.append(tmp)
            sample_count += 1
        sampledTraces = np.array(sampledTraces)
        result = np.reshape(sampledTraces, [batch_size * trace_length, 2])
        return result

    def generate_y(self, target_index):
        # print target_index
        output_size = self.config.output_size
        if target_index >= self.config.output_size:
            target_index = self.config.output_size - 1
        # tf_mod((self.target_out - output_layer) + 1.0, 2.0) - 1.0
        # print target_index, "target_index"
        result = [0.0] * self.config.output_size
        result[target_index] = 1.0
        # for i in range(self.config.output_size):
        #     # distance = abs(((abs(target_index - i) + output_size/2) % output_size) - output_size/2)
        #     diff1 = target_index - i
        #     diff2 = i - target_index
        #     distance = min((diff1 + output_size) % output_size, (diff2 + output_size) % output_size)
        #     if distance == 0:
        #         result[i] = 0.8
        #     elif distance == 1:
        #         result[i] = 0.1
        #     # elif distance == 2:
        #     #     result[i] = 0.05

        return [result]
