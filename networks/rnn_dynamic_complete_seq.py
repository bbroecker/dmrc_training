import copy
import inspect
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


def np_mod(x, y):
    return (x % y).astype(np.float32)


def modgrad(op, grad):
    x = op.inputs[
        0]  # the first argument (normally you need those to calculate the gradient, like the gradient of x^2 is 2x. )
    y = op.inputs[1]  # the second argument

    return grad * 1, grad * tf.negative(tf.floordiv(x, y))


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def tf_mod(x, y, name=None):
    with ops.op_scope([x, y], name, "mod") as name:
        z = py_func(np_mod,
                    [x, y],
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


class RRNDynamicModelCompleteSeq():
    def __init__(self, num_drones, config, is_training, log_path, weight_path, variable_scope, namespace=""):
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
        self.replay_buffer = ExperienceDualBuffer(config, num_drones, self.config.buffer_size)
        self.log_count = 0
        self.valid_counter = 0
        print "Variable scope {0}".format(variable_scope.name)
        save_var = {}
        # save_var[namespace + '_W_out'] = self.W_out
        # save_var[namespace + '_b_out'] = self.b_out
        #
        # for i in range(0, len(self.W)):
        #     save_var[namespace + '_W' + str(i)] = self.W[i]
        #     save_var[namespace + '_b' + str(i)] = self.b[i]
        summeries = []
        #
        # # Execute the LSTM cell here in any way, for example:
        # # Retrieve just the LSTM variables.
        save_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variable_scope.name)
        # for idx, lst in enumerate(lstm_variables):
        #     save_var[lst.key] = lst
        #     summeries.append(tf.summary.histogram('lst_' + str(idx), lst))

        with tf.name_scope(self.namespace + "_summaries"):
            summeries.append(tf.summary.histogram(namespace + "_weights_out", self.W_out))
            summeries.append(tf.summary.histogram(namespace + "_biases_out", self.b_out))
        if is_training:
            summeries.append(self.cost_sum)
            summeries.append(self.acc_sum)
            self.sum_merge = tf.summary.merge(summeries)
        self.saver = tf.train.Saver(save_var, max_to_keep=1)
        # self.saver = tf.train.Saver(max_to_keep=1)

    def load_weight(self, sess):
        print 'Loading Model...{0}'.format(self.weight_path)
        ckpt = tf.train.get_checkpoint_state(self.weight_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save_weights(self, sess, prefix):
        self.saver.save(sess, self.weight_path + '/model-' + str(prefix) + '.cptk')

    def save_weights_to_folder(self, sess, prefix, folder):
        print 'Saving Model...{0}'.format(folder)
        self.saver.save(sess, folder + '/model-' + str(prefix) + '.cptk')

    def data_type(self):
        return tf.float16 if self.config.use_fp16 else tf.float32

    def lstm_cell(self):
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is unfortunately not
        # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
        # an argument check here:
        acti = tf.tanh if self.config.use_tanh_rnn[0] else tf.nn.elu

        if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(
                self.config.hidden_size_rnn[0], forget_bias=self.config.forget_bias, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size, reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                self.config.hidden_size_rnn[0], forget_bias=self.config.forget_bias, state_is_tuple=True,
                activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size)

    def create_dense_layer(self, prev_layer, W, b, use_tanh):
        if use_tanh:
            layer = tf.nn.tanh(tf.add(tf.matmul(prev_layer, W), b))
        else:
            layer = tf.nn.elu(tf.add(tf.matmul(prev_layer, W), b))
        if self.is_training and self.config.keep_prob < 1:
            layer = tf.nn.dropout(layer, self.config.keep_prob)

        return layer


    def init_network(self):
        self.trace_length = tf.placeholder(dtype=tf.int32)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.tf_batch_size = tf.placeholder(dtype=tf.int32)
        # self.observations = tf.placeholder(tf.float32, [None, self.config.trace_length, self.config.state_count], name=self.namespace + "_observations")
        self.observations = tf.placeholder(tf.float32, [None, self.config.state_count],
                                           name=self.namespace + "_observations")

        # if self.is_training and self.config.keep_prob < 1:
        #     self.observations = tf.nn.dropout(self.observations, self.config.keep_prob)

        self.observations_rrn = tf.reshape(self.observations, [self.tf_batch_size, self.trace_length, self.config.state_count])

        attn_cell = self.lstm_cell
        if self.is_training and self.config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(self.lstm_cell(), output_keep_prob=self.config.keep_prob)

        self.stacked_lstm = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.config.num_layers)],
                                                        state_is_tuple=True)
        self.initial_state = self.stacked_lstm.zero_state(self.tf_batch_size, self.data_type())
        # self.initial_state = self.stacked_lstm.zero_state(self.config.batch_size, self.data_type())
        rnn_output, self.rnn_state = tf.nn.dynamic_rnn( \
            inputs=self.observations_rrn, cell=self.stacked_lstm, dtype=self.data_type(), initial_state=self.initial_state,
            scope=self.namespace + '_rnn')
        rnn_output = tf.reshape(rnn_output, shape=[-1, self.config.hidden_size_rnn[0]])

        dense_h_size = copy.copy(self.config.hidden_size_dense)
        dense_h_size.insert(0, self.config.hidden_size_rnn[-1])

        self.W_out = tf.get_variable(self.namespace + "_W_out", [dense_h_size[-1], 1], dtype=self.data_type(),
                                     trainable=self.is_training)
        self.b_out = tf.get_variable(self.namespace + "_b_out", [1, 1], dtype=self.data_type(),
                                     trainable=self.is_training)

        # rnn_output = tf.reshape(rnn_output, shape=[-1, self.config.hidden_size[0]])
        for i in range(1, len(dense_h_size)):
            self.W.append(tf.get_variable(self.namespace + "_W" + str(i), [dense_h_size[i - 1], dense_h_size[i]],
                                          dtype=self.data_type(), trainable=self.is_training))
            self.b.append(tf.get_variable(self.namespace + "_b" + str(i), [1, dense_h_size[i]], dtype=self.data_type(),
                                          trainable=self.is_training))

        # with tf.name_scope("input_layer"):
        layers = [self.create_dense_layer(rnn_output, self.W[0], self.b[0], self.config.use_tanh_dense[0])]
        for i in range(1, len(self.W)):
            # with tf.name_scope(self.namespace + "_layer" + str(i)):
            layers.append(self.create_dense_layer(layers[i - 1], self.W[i], self.b[i], self.config.use_tanh_dense[i]))

        # with tf.name_scope("output_layer"):
        self.output_layer = tf.add(tf.matmul(layers[-1], self.W_out), self.b_out)

        # self.output_layer = tf.add(tf.matmul(rnn_output[-1], self.W_out), self.b_out)

        self.predict = self.output_layer

        self.Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        # self.maskA = tf.zeros([self.tf_batch_size, self.trace_length // 2], name=self.namespace + "_Mask_A")
        # self.maskB = tf.ones([self.tf_batch_size, self.trace_length // 2], name=self.namespace + "_Mask_B")
        # self.mask = tf.concat([self.maskA, self.maskB], 1, name=self.namespace + "_Mask")
        # self.mask = tf.reshape(self.mask, [-1])
        # self.loss = tf.reduce_mean(self.td_error * self.mask, name=self.namespace + "_loss")
        # self.loss = tf.reduce_mean(self.td_error * self.mask, name=self.namespace + "_loss")
        # with tf.name_scope("accuracy"):
        #     correct_pred = tf.equal(tf.argmax(self.Y, 1), self.predict)  # Count correct predictions
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))  # Cast boolean to float to average
        #     # Add scalar summary for accuracy tensor
        #     self.acc_sum = tf.summary.scalar("accuracy", self.accuracy)
        #     validation = tf.summary.scalar("valid_accuracy", self.accuracy)
        #     test = []
        #     test.append(validation)
        #     self.sum_validation = tf.summary.merge(test)


        # self.td_error = tf.square(self.predict - self.Y)
        # self.accuracy = (2.0 - tf.abs(self.predict - self.Y)) / 2.0
        self.td_error = tf_mod((self.predict - self.Y) + 1.0, 2.0) - 1.0
        self.l = tf.square(self.td_error)
        self.accuracy = (1.0 - tf.abs(self.td_error)) / 1.0

        # self.td_error = tf.sqrt(tf.square(self.predict[0] - self.Y[0]) + tf.square(self.predict[1] - self.Y[1]))
        # self.accuracy = (2.0 - tf.abs(self.predict - self.Y)) / 2.0
        # self.accuracy = (2.0 - tf.abs(self.predict - self.Y)) / 2.8284
        # self.accuracy = tf.reduce_mean(a)
        self.loss = tf.reduce_mean(self.l)

        # self.accuracy = tf.reduce_mean(a)
        # self.loss = tf.reduce_mean(self.td_error)
        validation = self.variable_summaries(self.accuracy, "valid_accuracy")
        self.sum_validation = tf.summary.merge(validation)
        self.accuracy_mean = tf.reduce_mean(self.accuracy)

        if not self.is_training:
            return
        self.cost_sum = tf.summary.scalar("angle_cost_function", self.loss)
        self.acc_sum = tf.summary.scalar("angle_accuracy", tf.reduce_mean(self.accuracy_mean))
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
        self.replay_buffer = ExperienceDualBuffer(self.config, self.num_drones, self.buffer_size)

    def store_episodes(self):
        self.replay_buffer.add(self.episode_buffer.buffer)

    def gen_init_state(self, sess, batch_size):
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
            _ = sess.run([ self.updateModel], \
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


    def zero_state(self):
        return self.config.trace_length * [self.config.state_count * [0.0]]

    def validate(self, sess):
        # state_train = sess.run([self.initial_state], feed_dict={self.tf_batch_size: self.config.batch_size})

        state_train = sess.run([self.initial_state], feed_dict={self.tf_batch_size: self.config.valid_batch_size})
        trainBatch = self.replay_buffer.sample(self.config.valid_batch_size, self.config.trace_length)  # Get a random batch of experiences.
        x = trainBatch[:, 0]
        y = trainBatch[:, 1]

        # print "write stuff x: {0}".format(x)
        # print "write stuff y: {0}".format(y)
        acc, summary_str = sess.run([self.accuracy_mean, self.sum_validation], \
                                  feed_dict={self.observations: np.vstack(x),
                                             self.Y: np.vstack(y),\
                                             self.trace_length: self.config.trace_length,\
                                             self.initial_state: state_train, self.tf_batch_size: self.config.valid_batch_size})
        self.summary_writer.add_summary(summary_str, self.valid_counter)
        self.valid_counter += 1

        return acc

    def store_episode(self, state, y, bucket_id, seq_start):
        self.replay_buffer.add(state, y, bucket_id, seq_start)
        # self.replay_buffer.extend(
        #     np.reshape(np.array([state, angle]), [1, 2]))  # Save the experience to our episode buffer.

    def new_rrn_state(self, sess, state, state_rrn):
        return sess.run(self.rnn_state, \
                        feed_dict={self.observations: [state], self.trace_length: 1,
                                   self.initial_state: state_rrn, self.tf_batch_size: 1})


class ExperienceDualBuffer:
    def __init__(self, config, num_bucket, buffer_size):
        self.buffer_x = []
        self.buffer_y = []
        for n in range(num_bucket):
            self.buffer_x.append(deque(maxlen=buffer_size))
            self.buffer_y.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.config = config
        self.num_bucket = num_bucket

    def add(self, x, y, bucket_id, seq_start):
        self.buffer_x[bucket_id].append([x, seq_start])
        self.buffer_y[bucket_id].append(y)

    def sample(self, batch_size, trace_length):
        # sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        sample_count = 0
        while sample_count < batch_size:
            bucket = np.random.randint(0, self.num_bucket)
            point = np.random.randint(0, len(self.buffer_x[bucket]) + 1 - trace_length - 1)
            start_point = point
            for i in range(point, point + trace_length):
                if self.buffer_x[bucket][i][1]:
                    start_point = i
                    break

            if start_point + trace_length >= len(self.buffer_x[bucket]):
                continue
            tmp = []
            for i in range(point, point + trace_length):
                tmp.append([self.buffer_x[bucket][i][0], self.buffer_y[bucket][i]])
            # data1 = [self.buffer_x[i] for i in range(point, point + trace_length)]
            # data2 = [self.buffer_y[i] for i in range(point+1, point + trace_length + 1)]
            sampledTraces.append(tmp)
            sample_count += 1
        sampledTraces = np.array(sampledTraces)
        result = np.reshape(sampledTraces, [batch_size * trace_length, 2])
        return result


