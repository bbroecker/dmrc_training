import copy
import os
import time

import numpy as np
import tensorflow as tf
from src.networks import DQN_RNN

from tf_nn_sim.Environments.PuckNormalMulti.Puckworld_Environment import PuckNormalMultiworld


class MediumConfig(object):
    init_scale = 0.01
    # init_scale = 1.0
    max_learning_rate = 0.001
    min_learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 1
    # state_count = 5
    state_count = 3
    trace_length = 8
    mask = True
    hidden_size = 100
    max_epoch = 1
    # max_max_epoch = 39
    keep_prob = 1.0
    lr_decay = 0.8
    batch_size = 10
    use_tanh = False
    buffer_size = 50000
    forget_bias = 1.0
    use_fp16 = False
    valid_batch_size = 100
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_time = 70.0
    epsilon = 0.8
    gamma = 0.9
    store_every_ts = 1
    train_freq = 3
    clip = 1.0
    # vocab_size = 10000
    num_drones = 2
    pre_train_episodes = 2


def generate_path(src, cfg):
    output = src
    output += "lr" + str(cfg.max_learning_rate) + "_"
    if cfg.use_tanh:
        output += "tanh" + str(cfg.hidden_size) + "_"
    else:
        output += "relu" + str(cfg.hidden_size) + "_"
    output += "layers" + str(cfg.num_layers) + "_"
    output += "_batch" + str(cfg.batch_size)
    output += "_trace_length" + str(cfg.trace_length) + "_"
    output += "forget_bias_" + str(cfg.forget_bias)
    output += "_mask_" + str(cfg.mask)
    output += "_max_epoch_" + str(cfg.max_epoch)
    output += "_keep_prob_" + str(cfg.keep_prob)
    print output
    return output


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


# def predict_angle(network, env, rrn_state, state, cfg, angle=None):
#     pre, new_rrn_state = network.predict_angle(sess, state, rrn_state)
#     # print pre
#     pre = pre[0]
#     if angle is not None:
#         tmp1 = (pre - angle + cfg.output_size) % cfg.output_size
#         tmp2 = (pre - angle + cfg.output_size) % cfg.output_size
#         diff = min(tmp1, tmp2)
#         average_error.append(1.0 if diff <= 1 else 0.0)
#
#     # average_error.append(abs(angle - pre))
#     # pre = float(translate(pre, 0, cfg.output_size, 0.0, 1.0))
#     pre = float(translate(pre, 0, cfg.output_size, -1.0, 1.0))
#     # print pre
#     # env.angle_obstacle[0] = pre * np.pi
#     env.angle_obstacle[0] = pre * np.pi
#     test = 0
#     while env.angle_obstacle[0] > np.pi and env.angle_obstacle[0] is not None:
#         if test > 100:
#             env.angle_obstacle[0] = None
#             break
#         env.angle_obstacle[0] -= 2 * np.pi
#         test += 1
#     test = 0
#     while env.angle_obstacle[0] < -np.pi and env.angle_obstacle[0] is not None:
#         if test > 100:
#             env.angle_obstacle[0] = None
#             break
#         env.angle_obstacle[0] += 2 * np.pi
#         test += 1
#     return new_rrn_state


def create_nn_folders(weight_path, log_path, angle_load):
    gen_weight_path = weight_path
    gen_log_path = log_path

    if not angle_load:
        gen_weight_path = generate_path(weight_path, cfg)
        gen_log_path = generate_path(log_path, cfg)
        i = 0
        while os.path.exists(gen_log_path + "_" + str(i)):
            i += 1
        gen_weight_path += "_" + str(i)
        gen_log_path += "_" + str(i)

    # Make a path for our model to be saved in.
    if not os.path.exists(gen_log_path):
        os.makedirs(gen_log_path)
    if not os.path.exists(gen_weight_path):
        os.makedirs(gen_weight_path)

    return gen_weight_path, gen_log_path


def trigger_movement_nn(sess, episode_count, step_count, env, states, rrn_states, training_network, predict_network,
                        learning_rate, cfg, total_steps):
    actions = [0] * env.num_drones
    for d_id in range(env.num_drones):
        if np.random.rand(1) < cfg.epsilon or episode_count < cfg.pre_train_episodes:
            rrn_states[d_id] = predict_network.new_rrn_state(sess, rrn_states[d_id], states[d_id])
            actions[d_id] = np.random.randint(0, env.num_actions)
            # a = 5
        else:
            a, rrn_states[d_id] = predict_network.predict_action(sess, rrn_states[d_id], states[d_id])
            actions[d_id] = a[0]

    s1, r, d = env.step(actions)
    for d_id in range(env.num_drones):
        training_network.store_episode(states[d_id], actions[d_id], r[d_id], s1[d_id], d[d_id], d_id)
    debug_data = [None] * env.num_drones
    if episode_count > 3 or load_network:
        if env.frontend.call_count % env.frontend.skip == 0:
            for d_id in range(env.num_drones):
                predict, q_out, td_error = training_network.single_train_network(sess, learning_rate, rrn_states[d_id], states[d_id], actions[d_id],
                                                                         r[d_id], s1[d_id], d[d_id])
                debug_data[d_id] = {"predict": predict, "q_values": q_out, "td_error": td_error}
    if episode_count > 3 and total_steps % cfg.train_freq == 0 and len(
            training_network.replay_buffer.buffer[0]) > cfg.batch_size * cfg.trace_length:
        training_network.train_network(sess, learning_rate, True if total_steps % (cfg.train_freq * 5) == 0 else False)

    done = True in d
    if episode_count > 5 and (done or total_steps % 2000 == 0):
        training_network.save_weights(sess, total_steps)

    return s1, r, rrn_states, done, debug_data


if __name__ == '__main__':

    # env = Puckworld()
    env = PuckNormalMultiworld()
    # pre_train_steps = 10000 #How many steps of random actions before training begins.
    # create lists to contain total rewards and steps per episode
    load_network = True
    # load_network = False
    max_time = 960
    # log_path = "./log/rnn_dqn/multi/batch2/"  # The path to save our model to.
    log_path = "./log/rnn_dqn/multi/batch2/lr0.001_relu100_layers1__batch10_trace_length8_forget_bias_1.0_mask_True_max_epoch_1_keep_prob_1.0_2/"  # The path to save our model to.
    # weight_path = "./weights_save/rnn_dqn/multi/batch2/"  # The path to save our model to.
    weight_path = "./weights_save/rnn_dqn/multi/batch2/lr0.001_relu100_layers1__batch10_trace_length8_forget_bias_1.0_mask_True_max_epoch_1_keep_prob_1.0_2/"  # The path to save our model to.

    jList = []
    rList = []
    total_steps = 0
    # cfg = MediumConfig()
    for cfg in [MediumConfig()]:
        predict_cfg = copy.copy(cfg)
        predict_cfg.batch_size = 1

        learn_rate_range = cfg.max_learning_rate - cfg.min_learning_rate
        epsilon_range = cfg.epsilon_start - cfg.epsilon_end
        percent = 0.0
        epsi_per = 0.0
        with tf.Graph().as_default():
            initializer = tf.random_normal_initializer(stddev=cfg.init_scale)
            gen_weight_path, gen_log_path = create_nn_folders(weight_path, log_path, load_network)
            # with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                training_network = DQN_RNN(env=env, config=cfg, namespace="DQN", is_training=True,
                                           log_path=gen_log_path, weight_path=gen_weight_path)

            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                predict_network = DQN_RNN(env=env, config=cfg, namespace="DQN", is_training=False,
                                          log_path=gen_log_path, weight_path=gen_weight_path)
                predict_network.replay_buffer = training_network.replay_buffer

            start_time = time.time()

            keep = 0.0
            # sv = tf.train.Supervisor(logdir=angle_path)
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                training_network.init_logger(sess)
                if load_network:
                    print('Loading Angle Model...')
                    training_network.load_weight(sess)
                # for i in range(env.num_episodes):
                episode_count = 0
                while True:

                    s = env.reset()
                    d = False
                    rAll = 0
                    step_count = 0
                    rrn_states = []
                    a = [0] * env.num_drones
                    for r in range(env.num_drones):
                        rrn_states.append(predict_network.gen_init_state(sess, 1))
                    rewards = []
                    for step_count in range(
                            env.max_epLength):  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                        current_learn_rate = cfg.max_learning_rate - learn_rate_range * percent
                        cfg.epsilon = cfg.epsilon_start - epsilon_range * epsi_per
                        s1, r, rrn_states, done, debug_data = trigger_movement_nn(sess, episode_count, step_count, env,
                                                                                  s, rrn_states,
                                                                                  training_network, predict_network,
                                                                                  current_learn_rate, cfg, total_steps)
                        # s1, r, rrn_states, done, debug_data
                        s = s1
                        rewards.extend(r)
                        env.render(debug_data)
                        total_steps += 1
                        # if done:
                        #     break
                    # predict_network.validate(sess, summary_writer)
                    training_network.store_rewards(sess, rewards, episode_count)
                    episode_count += 1
                    percent = (time.time() - start_time) / (max_time * 60)
                    epsi_per = (time.time() - start_time) / (cfg.epsilon_time * 60)
                    epsi_per = 1.0 if epsi_per > 1.0 else epsi_per
                    current_learn_rate = cfg.max_learning_rate - learn_rate_range * percent
                    print "percent done: {0}% learn rate {1} epsilon {2}".format(percent * 100, current_learn_rate,
                                                                                 cfg.epsilon)
                    if percent >= 1.0:
                        break
                training_network.save_weights(sess, total_steps)
                # print("Percent of succesful episodes: " + str(sum(rList) / env.num_episodes) + "%")
