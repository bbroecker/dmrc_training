import copy
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
    init_scale = 0.04
    # init_scale = 1.0
    max_learning_rate = 0.1
    min_learning_rate = 0.05
    max_grad_norm = 5
    # state_count = 5
    state_count = 3
    trace_length = 10
    hidden_size = [1024, 512, 256]
    output_size = 1
    # max_epoch = 6
    # max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 60
    use_tanh = [True, True, True]
    buffer_size = 500000
    forget_bias = 0.0
    use_fp16 = False
    # vocab_size = 10000



class GeneralCNNBatchModel():
    def __init__(self, config, is_training, namespace = ""):
        self.config = config
        self.is_training = is_training
        self.namespace = namespace
        self.layers = []
        self.W = []
        self.b = []
        self.W_out = None
        self.init_network()
        self.replay_buffer = ExperienceDualBuffer(self.config.buffer_size, config)
        self.log_count = 0
        self.valid_counter = 0
        save_var = {}
        # save_var[namespace + '_W_out'] = self.W_out
        # save_var[namespace + '_b_out'] = self.b_out
        summeries = []

        # for i in range(0, len(self.W)):
        #     save_var[namespace + '_W' + str(i)] = self.W[i]
        #     save_var[namespace + '_b' + str(i)] = self.b[i]
        #     with tf.name_scope(self.namespace + "_weight_summaries"):
        #         summeries.append(tf.summary.histogram(namespace + "_weights" + str(i), self.W[i]))
        #     with tf.name_scope(self.namespace + "_biases_summaries"):
        #         summeries.append(tf.summary.histogram(namespace + "_biases" + str(i), self.b[i]))
        #
        # for i in range(0, len(self.layers)):
        #     with tf.name_scope(self.namespace + "_activation_summaries"):
        #         summeries.append(tf.summary.histogram(namespace + "_activation" + str(i), self.layers[i]))


        # with tf.name_scope(self.namespace + "_summaries"):
        #     summeries.append(tf.summary.histogram(namespace + "_weights_out", self.W_out))
        #     summeries.append(tf.summary.histogram(namespace + "_biases_out", self.b_out))
        if is_training:
            # pass
            summeries.append(self.cost_sum)
            summeries.append(self.acc_sum)
            self.sum_merge = tf.summary.merge(summeries)
        # self.saver = tf.train.Saver(save_var)


    def load_weight(self, sess, path):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def save_weights(self, sess, path, prefix):
        # self.saver.save(sess, path + '/model-' + str(prefix) + '.cptk')
        # self.saver.save(sess, path + '/model-' + str(prefix) + '.cptk')
        pass

    def data_type(self):
        return tf.float16 if self.config.use_fp16 else tf.float32

    def lstm_cell(self):
        # With the latest TensorFlow source code (as of Mar 27, 2017),
        # the BasicLSTMCell will need a reuse parameter which is unfortunately not
        # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
        # an argument check here:
        acti = tf.tanh if self.config.use_tanh else tf.nn.relu

        if 'reuse' in inspect.getargspec(
                tf.contrib.rnn.BasicLSTMCell.__init__).args:
            return tf.contrib.rnn.BasicLSTMCell(
                self.config.hidden_size, forget_bias=self.config.forget_bias, state_is_tuple=True,
                reuse=tf.get_variable_scope().reuse, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size, reuse=tf.get_variable_scope().reuse)
        else:
            return tf.contrib.rnn.BasicLSTMCell(
                self.config.hidden_size, forget_bias=self.config.forget_bias, state_is_tuple=True, activation=acti)
            # return tf.contrib.rnn.GRUCell(
            #     self.config.hidden_size)

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
        if self.is_training and self.config.keep_prob < 1:
            layer = tf.nn.dropout(layer, self.config.keep_prob)

        return layer

    def dense(self, x, size, scope):
        return tf.contrib.layers.fully_connected(x, size,
                                                 activation_fn=None,
                                                 scope=scope)

    def dense_batch_relu_tanh(self, layer, hidden, phase, scope, use_tanh):
        with tf.variable_scope(scope):
            h1 = tf.contrib.layers.fully_connected(layer, hidden,
                                                   activation_fn=None,
                                                   scope='dense')
            h2 = tf.contrib.layers.batch_norm(h1,
                                              center=True, scale=True,
                                              is_training=phase,
                                              scope='bn')
            if use_tanh:
                layer = tf.nn.tanh(h2, 'tanh')
            else:
                layer = tf.nn.relu(h2, 'relu')

            if self.is_training and self.config.keep_prob < 1:
                layer = tf.nn.dropout(layer, self.config.keep_prob)

            return layer


    def init_network(self):
        h_size = copy.copy(self.config.hidden_size)
        h_size.insert(0, self.config.state_count * self.config.trace_length)
        self.trace_length = tf.placeholder(dtype=tf.int32)
        self.Y = tf.placeholder(tf.float32, [None, self.config.output_size], name="Y")
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        # self.tf_batch_size = tf.placeholder(dtype=tf.int32)
        self.observations = tf.placeholder(tf.float32, [None, self.config.state_count * self.config.trace_length], name=self.namespace + "_observations")
        if self.is_training and self.config.keep_prob < 1:
            self.observations = tf.nn.dropout(self.observations, self.config.keep_prob)


        # with tf.name_scope("input_layer"):
        self.layers = [self.dense_batch_relu_tanh(self.observations, h_size[0], self.is_training, "input_layer", self.config.use_tanh[0])]
        # self.layers = [self.create_layer(prev_layer=self.observations, W=self.W[0], b=None, use_tanh=self.config.use_tanh[0])]
        for i in range(1, len(h_size)):
            # with tf.name_scope(self.namespace + "_layer" + str(i)):
            self.layers.append(self.dense_batch_relu_tanh(self.layers[i-1], h_size[i], self.is_training, "hidden_layers" + str(i), self.config.use_tanh[i-1]))
            # self.layers.append(self.create_layer(prev_layer=self.layers[i-1], W=self.W[i], b=None, use_tanh=self.config.use_tanh[i]))

        # with tf.name_scope("output_layer"):
        # output_layer = tf.add(tf.matmul(self.layers[-1], self.W_out), self.b_out)
        output_layer = self.dense(self.layers[-1], self.config.output_size, "output_layer")
        # output_layer = tf.matmul(self.layers[-1], self.W_out)

        self.predict = tf.argmax(output_layer, 1)



        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.Y, 1), self.predict)  # Count correct predictions
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))  # Cast boolean to float to average
            # Add scalar summary for accuracy tensor
            self.acc_sum = tf.summary.scalar("accuracy", self.accuracy)
            validation = tf.summary.scalar("valid_accuracy", self.accuracy)
            test = []
            test.append(validation)
            self.sum_validation = tf.summary.merge(test)

        if not self.is_training:
            return

        # self.target_out = tf.placeholder(shape=[None, 1], dtype=tf.int32)
        # self.td_error = tf.square(self.target_angle - self.angle_out)
        # tmp = tf_mod((self.target_out - output_layer) + 1.0, 2.0) - 1.0
        # tmp = self.target_out - tf.cast(self.predict, tf.int32)
        # self.td_error = tf.square(tmp)
        # self.labels = tf.one_hot(self.target_out, self.config.output_size, dtype=tf.float32)
        # cost = tf.reduce_sum(tf.pow(self.target_angle - self.angle_out, 2)) / (n_observations - 1)
        # self.td_clamp1 = tf.minimum(self.td_error, 1.0)
        # self.td_clamp2 = tf.maximum(self.td_clamp1, -1.0)
        # self.accuracy = tf.reduce_mean((1.0 - tf.cast(tf.abs(tmp), tf.float32) / self.config.output_size))
        # self.loss = tf.reduce_mean(self.td_error, name=self.namespace + "_loss")
        # self.td_clamp1 = tf.minimum(self.td_error, 1.0)
        # self.td_clamp2 = tf.maximum(self.td_clamp1, -1.0)

        # self.maskA = tf.zeros([self.tf_batch_size, self.trace_length // 2], name=self.namespace + "_Mask_A")
        # self.maskB = tf.ones([self.tf_batch_size, self.trace_length // 2], name=self.namespace + "_Mask_B")
        # self.mask = tf.concat([self.maskA, self.maskB], 1, name=self.namespace + "_Mask")
        # self.mask = tf.reshape(self.mask, [-1])
        # self.loss = tf.reduce_mean(self.td_error * self.mask, name=self.namespace + "_loss")
        # self.loss = tf.reduce_mean(self.td_error * self.mask, name=self.namespace + "_loss")

        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
        #                                   self.config.max_grad_norm)
        # tmp = tf.cast(self.predict, "float") - tf.cast(tf.argmax(self.Y), "float")
        # self.td_error = tf.square(tmp)
        # self.labels = tf.one_hot(self.target_out, self.config.output_size, dtype=tf.float32)
        # self.loss = tf.reduce_sum(self.td_error)

        # self.loss = tf.reduce_sum(tf.square(output_layer - self.Y))

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=self.Y, name=None))
        self.cost_sum = tf.summary.scalar("angle_cost_function", self.loss)
        self.acc_sum = tf.summary.scalar("angle_accuracy", self.accuracy)
        self._lr = tf.Variable(0.0, trainable=False)

        # self.trainer = tf.train.RMSPropOptimizer(self._lr, 0.9)
        # self.trainer = tf.train.GradientDescentOptimizer(self._lr)
        self.trainer = tf.train.AdamOptimizer(self._lr)
        # self.updateModel = optimizer.apply_gradients(
        #     zip(grads, tvars),
        #     global_step=tf.contrib.framework.get_or_create_global_step())
        self.updateModel = self.trainer.minimize(self.loss)


        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        # self.trainer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)
        # self.updateModel = self.trainer.minimize(self.loss)

    def reset_buffer(self):
        self.replay_buffer = ExperienceDualBuffer(self.buffer_size)

    def store_episodes(self):
        self.replay_buffer.add(self.episode_buffer.buffer)

    def gen_init_state(self):
        return [0.0] * self.config.trace_length * self.config.state_count

    def train_network(self, sess, learn_rate, epoch, summary_writer = None):
        sess.run(self._lr_update, feed_dict={self._new_lr: learn_rate})
        trainBatch = self.replay_buffer.sample(self.config.batch_size, self.config.trace_length)  # Get a random batch of experiences.
        x = trainBatch[:, 0]
        y = trainBatch[:, 1]
        loss = 0
        for run in range(epoch):
            if summary_writer is not None:
                # print "write stuff x: {0}".format(x)
                # print "write stuff y: {0}".format(y)
                summary_str, _, loss = sess.run([self.sum_merge, self.updateModel, self.loss], \
                                          feed_dict={self.observations: np.vstack(x),
                                                     self.Y: np.vstack(y)})
                summary_writer.add_summary(summary_str, self.log_count)
                self.log_count += 1
            else:
                _, loss = sess.run([ self.updateModel, self.loss], \
                             feed_dict={self.observations: np.vstack(x),
                                        self.Y: np.vstack(y)})
        return loss

    def validate(self, sess, summary_writer):
        b = min(len(self.replay_buffer.buffer_x)-self.config.trace_length, self.config.valid_batch_size)
        trainBatch = self.replay_buffer.sample(b, self.config.trace_length)  # Get a random batch of experiences.
        x = trainBatch[:, 0]
        y = trainBatch[:, 1]
        loss = 0
        # print "write stuff x: {0}".format(x)
        # print "write stuff y: {0}".format(y)
        summary_str = sess.run(self.sum_validation, \
                                  feed_dict={self.observations: np.vstack(x),
                                             self.Y: np.vstack(y)})
        summary_writer.add_summary(summary_str, self.valid_counter)
        self.valid_counter += 1


        return loss




    def predict_angle(self, sess, current_state, state_history):
        tmp = copy.copy(state_history)
        del tmp[0: self.config.state_count]
        tmp.extend(current_state)
        a = sess.run([self.predict], \
                             feed_dict={self.observations: [tmp]})
        return a, tmp

    def store_episode(self, state, y):
        self.replay_buffer.add(state, y)
        # self.replay_buffer.extend(
        #     np.reshape(np.array([state, angle]), [1, 2]))  # Save the experience to our episode buffer.

    def zero_state(self):
        return [0.0] * self.config.trace_length * self.config.state_count

class ExperienceDualBuffer:
    def __init__(self, buffer_size, config):
        self.config = config
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, x, y):
        self.buffer_x.append(x)
        self.buffer_y.append(y)

    def sample(self, batch_size, trace_length):
        # sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        sample_count = 0
        while sample_count < batch_size:
            point = np.random.randint(0, len(self.buffer_x) + 1 - trace_length - 1)
            tmp = []
            for i in range(point, point + trace_length):
                tmp.extend(self.buffer_x[i])
            # data1 = [self.buffer_x[i] for i in range(point, point + trace_length)]
            # data2 = [self.buffer_y[i] for i in range(point+1, point + trace_length + 1)]
            sampledTraces.append([tmp, self.generate_y(self.buffer_y[point + trace_length])])
            sample_count += 1
        sampledTraces = np.array(sampledTraces)
        result = np.reshape(sampledTraces, [batch_size, 2])
        return result

    def generate_y(self, target_index):
        # print target_index
        output_size = self.config.output_size
        # tf_mod((self.target_out - output_layer) + 1.0, 2.0) - 1.0
        result = [0.0] * self.config.output_size
        result[target_index] = 1.0
        for i in range(self.config.output_size):
            # distance = abs(((abs(target_index - i) + output_size/2) % output_size) - output_size/2)
            diff1 = target_index - i
            diff2 = i - target_index
            distance = min((diff1 + output_size) % output_size, (diff2 + output_size) % output_size)
            if distance == 0:
                result[i] = 0.8
            elif distance == 1:
                result[i] = 0.1
            # elif distance == 2:
            #     result[i] = 0.05

        return [result]