import copy
import os
import threading
import time

import numpy as np
import tensorflow as tf
import collections

from Config.NNConfigAngle import NNConfigAngle
from Config.NNConfigMovement import NNConfigMovement
from Config.SimConfig import SimulationConfig
from Environments.PuckRealMultiFinal.Puckworld_Environment import PuckRealMultiworldFinal
from networks.dqn import DQN
from networks.rrn_cnn_multitask_join import GeneralRRNDiscreteModelMultitaskJointLoss
from networks.rrn_dqn import DQN_RNN
from main.helper import translate, create_nn_folders

FRONTEND_ACTIVE = True


def discrete_to_angle(discrete_angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    return step_size * discrete_angle - np.pi


def angle_to_discrete(angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    index = int(round((angle + np.pi) / step_size))
    return index


def trigger_angle_nn(sess, predict_network, env, rrn_states_goal_angle,
                     rrn_states_obstacle_angle, cfg, episode_count):
    env.predict_obstacle_angle = [0.0] * env.num_drones
    env.predict_goal_angle = [0.0] * env.num_drones
    env.predict_obstacle_orientation = {}
    predict_angle_obs = [0.0] * env.num_drones
    predict_orientation_obs = [0.0] * env.num_drones
    predict_angle_goal = [0.0] * env.num_drones
    predict_orientation_goal = [0.0] * env.num_drones
    error = [0.0] * env.num_drones
    error_orientation = [0.0] * env.num_drones
    for d_id in range(env.num_drones):
        c_id = env.closes_drone(d_id)
        if env.training_goal:
            goal_angle = env.get_goal_angle_train(d_id)
            # state_goal = env.get_train_goal_angle_state(d_id, cfg.min_x, cfg.max_x, distance_mean=0.0,
            #                                             distance_variance=0.0312979988458)
            state_goal = env.get_train_goal_angle_state(d_id, cfg.min_x, cfg.max_x)
        else:
            goal_angle = env.get_goal_angle(d_id)
            # state_goal = env.get_goal_angle_state(d_id, cfg.min_x, cfg.max_x, distance_mean=0.0,
            #                                       distance_variance=0.0312979988458)
            state_goal = env.get_goal_angle_state(d_id, cfg.min_x, cfg.max_x)
        obs_angle = env.get_obstacle_angle(d_id)
        # angle = int(translate(angle, -1.0, 1.0, 0, cfg.output_size - 0.001))
        state_obs = env.get_obstacle_angle_state(d_id, cfg.min_x, cfg.max_x)
        # state_obs = env.get_obstacle_angle_state(d_id, cfg.min_x, cfg.max_x, distance_mean=0.0,
        #                                          distance_variance=0.0312979988458)

        orientation_diff_obs = wrap_angle(env.orientation[c_id] - env.orientation[d_id])
        orientation_diff_goal = 0.0
        # orientation_diff = wrap_angle(env.orientation[c_id] - env.orientation[d_id])
        angle_training_network.store_episode(state_obs, angle_to_discrete(obs_angle, cfg),
                                             angle_to_discrete(orientation_diff_obs, cfg), d_id, 0, episode_count)
        goal_id = episode_count if env.training_goal else env.goal_id[d_id]

        angle_training_network.store_episode(state_goal, angle_to_discrete(goal_angle, cfg),
                                             angle_to_discrete(orientation_diff_goal, cfg), d_id, 1, goal_id)
        # return a, o, state1, state2
        predict_angle_obs[d_id], predict_orientation_obs[d_id], rrn_states_obstacle_angle[
            d_id] = predict_network.predict_angle_orientation(sess, state_obs,
                                                              rrn_states_obstacle_angle[d_id])
        predict_angle_goal[d_id], predict_orientation_goal[d_id], rrn_states_goal_angle[
            d_id] = predict_network.predict_angle_orientation(sess, state_goal,
                                                              rrn_states_goal_angle[d_id])
        # print "goal {} predict {}".format(angle_to_discrete(goal_angle, cfg), predict_angle_goal[d_id])
        er1 = 0.0 if abs(angle_to_discrete(goal_angle, cfg) - predict_angle_goal[d_id]) < 2 else 1.0
        # print goal_angle, predict_angle_goal[d_id]
        er2 = 0.0 if abs(angle_to_discrete(obs_angle, cfg) - predict_angle_obs[d_id]) < 2 else 1.0
        # print "obs {} predict {}".format(angle_to_discrete(obs_angle, cfg), predict_angle_obs[d_id])
        # print obs_angle, predict_angle_obs[d_id]

        error[d_id] = (er1 + er2) / 2.0
        env.predict_obstacle_orientation[d_id] = discrete_to_angle(predict_orientation_obs[d_id], cfg)
        er1 = 0.0 if abs(angle_to_discrete(orientation_diff_obs, cfg) - predict_orientation_obs[d_id]) < 2 else 1.0
        er2 = 0.0 if abs(angle_to_discrete(orientation_diff_goal, cfg) - predict_orientation_goal[d_id]) < 2 else 1.0
        # print "should {} is: {}".format(angle_to_discrete(orientation_diff_obs, cfg), predict_orientation_obs[d_id])
        error_orientation[d_id] = (er1 + er2) / 2.0

        predict_angle_obs[d_id] = discrete_to_angle(predict_angle_obs[d_id], cfg)
        predict_angle_goal[d_id] = discrete_to_angle(predict_angle_goal[d_id], cfg)
        goal_angle = discrete_to_angle(angle_to_discrete(goal_angle, cfg), cfg)
        obs_angle = discrete_to_angle(angle_to_discrete(obs_angle, cfg), cfg)

        predict_angle_obs[d_id] = wrap_angle(predict_angle_obs[d_id])
        predict_angle_goal[d_id] = wrap_angle(predict_angle_goal[d_id])
        env.predict_obstacle_angle[d_id] = predict_angle_obs[d_id]
        # env.predict_obstacle_angle[d_id] = obs_angle
        env.predict_goal_angle[d_id] = predict_angle_goal[d_id]
        # env.predict_goal_angle[d_id] = goal_angle

    return rrn_states_goal_angle, rrn_states_obstacle_angle, predict_angle_obs, predict_angle_goal, error, error_orientation


def wrap_angle(angle):
    test = 0
    while angle > np.pi:
        if test > 100:
            angle = None
            break
        angle -= 2 * np.pi
        test += 1
    test = 0
    while angle < -np.pi:
        if test > 100:
            angle = None
            break
        angle += 2 * np.pi
        test += 1
    return angle


def trigger_movement_nn(sess, env, states, predict_network, cfg):
    actions = [0] * env.num_drones
    for d_id in range(env.num_drones):
        a = predict_network.predict_action(sess, states[d_id])
        actions[d_id] = a[0][0]
        # actions[d_id] = 0
        # print "action {0} id {1}".format(a[0][0], d_id)
        # if d_id == 0:
        #     actions[d_id] = 9
    goal_collect, s1, r, crash = env.step(actions)
    # training_network.store_episode(history_states[d_id], actions[d_id], r[d_id], new_history[d_id], d[d_id], d_id)
    debug_data = [None] * env.num_drones

    return s1, actions, r, crash, debug_data, goal_collect



CURRENT_BEST_ACCURACY = 0.0


class TrainingWorker(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.is_training = False
        self.accuracy = 0.0

    def start_training(self, sess, env, training_network, predict_network, cfg):
        self.accuracy = 0.0
        self.is_training = True
        self.sess = sess
        self.env = env
        self.training_network = training_network
        self.predict_network = predict_network
        self.cfg = cfg
        self.pause = False
        self.start()

    def resume(self):
        self.pause = False

    def run(self):
        steps = 0
        print "START TRAINING"
        start_time = time.time()
        learn_rate_range = self.cfg.max_learning_rate - self.cfg.min_learning_rate
        while self.is_training:
            if self.pause:
                continue
            learn_rate_per = (time.time() - start_time) / (self.cfg.learning_rate_time * 60)
            learn_rate_per = 1.0 if learn_rate_per > 1.0 else learn_rate_per
            learn_rate = self.cfg.max_learning_rate - learn_rate_range * learn_rate_per
            self.training_network.train_network(self.sess, learn_rate,
                                                True if steps % self.cfg.logging_freq == 0 else False)
            if steps % 100 == 0:
                self.accuracy, a2 = self.predict_network.validate(sess)
                print "valid accuracy {}".format(a2)
            self.pause = True
            steps += 1

    def stop_training(self):
        self.is_training = False


def use_predict_angle(env, s1, r, pre_obs, pre_goal, error, use_accurace_reward):
    for d_id in range(env.num_drones):
        o_distance = env.get_obstacle_distance(d_id)
        o_dx = translate(np.cos(pre_obs[d_id]) * o_distance,
                         -movement_cfg_predict.max_sensor_distance,
                         movement_cfg_predict.max_sensor_distance, angle_cfg_predict.min_x,
                         angle_cfg_predict.max_x)
        o_dy = translate(np.sin(pre_obs[d_id]) * o_distance,
                         -movement_cfg_predict.max_sensor_distance,
                         movement_cfg_predict.max_sensor_distance, angle_cfg_predict.min_x,
                         angle_cfg_predict.max_x)
        g_distance = env.get_goal_distance(d_id)
        g_dx = translate(np.cos(pre_goal[d_id]) * (g_distance),
                         -movement_cfg_predict.max_sensor_distance,
                         movement_cfg_predict.max_sensor_distance, angle_cfg_predict.min_x,
                         angle_cfg_predict.max_x)
        g_dy = translate(np.sin(pre_goal[d_id]) * (g_distance),
                         -movement_cfg_predict.max_sensor_distance,
                         movement_cfg_predict.max_sensor_distance, angle_cfg_predict.min_x,
                         angle_cfg_predict.max_x)
        if use_accurace_reward:
            r[d_id] -= error[d_id]
        # state = [pvx, pvy, dx, dy, o_dx, o_dy, update_tick]
        s1[d_id][2] = g_dx
        s1[d_id][3] = g_dy
        s1[d_id][4] = o_dx
        s1[d_id][5] = o_dy
    return s1, r


class AngleConfig1(object):
    # init_scale = 0.01
    init_scale = 0.04419
    # init_scale = 1.0
    init_constant = 0.1
    max_learning_rate = 0.0004
    # max_learning_rate = 0.001
    min_learning_rate = 0.0004
    # min_learning_rate = 0.001
    learning_rate_time = 120.0
    max_grad_norm = 30
    num_layers = 1
    state_count = 6
    # state_count = 3
    trace_length = 40
    min_history = 30
    buffer_size = 50000
    update_length = 10
    accuracy_length = 10
    hidden_size_rnn = [512, 512]
    hidden_size_dense = []
    # max_epoch = 6
    # max_max_epoch = 39
    keep_prob = 0.8
    lr_decay = 0.8
    batch_size = 30
    use_tanh_rnn = [True, True]
    use_tanh_dense = []
    forget_bias = 0.0
    use_fp16 = False
    output_size = 16
    valid_batch_size = 300
    log_folder = None
    logging_freq = 220.0
    # weight_folder = None
    max_y = 1.0
    min_y = -1.0
    max_x = 1.0
    min_x = -1.0
    # weight_folder = "/home/broecker/src/cf_nn_weights/rrn_dynamic/predict_angle/angle2/weights"
    weight_folder = None
    t_bucketsize = [200, 200]
    bucket_types = 2
    # vocab_size = 10000
    # optimizer = "GradientDescentOptimizer"
    # optimizer = "AdamOptimizer"
    optimizer = "RMSPropOptimizer"
    best_result = 0.70

class AngleConfig2(object):
    # init_scale = 0.01
    init_scale = 0.04419
    # init_scale = 1.0
    init_constant = 0.1
    max_learning_rate = 0.0004
    # max_learning_rate = 0.001
    min_learning_rate = 0.0004
    # min_learning_rate = 0.001
    learning_rate_time = 120.0
    max_grad_norm = 30
    num_layers = 1
    state_count = 6
    # state_count = 3
    trace_length = 40
    min_history = 30
    buffer_size = 50000
    update_length = 5
    accuracy_length = 10
    hidden_size_rnn = [512, 512]
    hidden_size_dense = []
    # max_epoch = 6
    # max_max_epoch = 39
    keep_prob = 0.8
    lr_decay = 0.8
    batch_size = 30
    use_tanh_rnn = [True, True]
    use_tanh_dense = []
    forget_bias = 0.0
    use_fp16 = False
    output_size = 16
    valid_batch_size = 300
    log_folder = None
    logging_freq = 220.0
    # weight_folder = None
    max_y = 1.0
    min_y = -1.0
    max_x = 1.0
    min_x = -1.0
    # weight_folder = "/home/broecker/src/cf_nn_weights/rrn_dynamic/predict_angle/angle2/weights"
    weight_folder = None
    t_bucketsize = [200, 200]
    bucket_types = 2
    # vocab_size = 10000
    # optimizer = "GradientDescentOptimizer"
    # optimizer = "AdamOptimizer"
    optimizer = "RMSPropOptimizer"
    best_result = 0.70

class AngleConfig3(object):
    # init_scale = 0.01
    init_scale = 0.04419
    # init_scale = 1.0
    init_constant = 0.1
    max_learning_rate = 0.0004
    # max_learning_rate = 0.001
    min_learning_rate = 0.0004
    # min_learning_rate = 0.001
    learning_rate_time = 120.0
    max_grad_norm = 30
    num_layers = 1
    state_count = 6
    # state_count = 3
    trace_length = 40
    min_history = 30
    buffer_size = 50000
    update_length = 1
    accuracy_length = 10
    hidden_size_rnn = [512, 512]
    hidden_size_dense = []
    # max_epoch = 6
    # max_max_epoch = 39
    keep_prob = 0.8
    lr_decay = 0.8
    batch_size = 30
    use_tanh_rnn = [True, True]
    use_tanh_dense = []
    forget_bias = 0.0
    use_fp16 = False
    output_size = 16
    valid_batch_size = 300
    log_folder = None
    logging_freq = 220.0
    # weight_folder = None
    max_y = 1.0
    min_y = -1.0
    max_x = 1.0
    min_x = -1.0
    # weight_folder = "/home/broecker/src/cf_nn_weights/rrn_dynamic/predict_angle/angle2/weights"
    weight_folder = None
    t_bucketsize = [200, 200]
    bucket_types = 2
    # vocab_size = 10000
    # optimizer = "GradientDescentOptimizer"
    # optimizer = "AdamOptimizer"
    optimizer = "RMSPropOptimizer"
    best_result = 0.70

if __name__ == '__main__':

    # env = Puckworld()
    # env = PuckNormalMultiworld()
    sim_cfg = SimulationConfig("../Config/Simulation/sim_config.yaml")
    # movement_cfg = NNConfigMovement("/home/broecker/src/cf_nn_weights/dqn/drqn_1/nn_config_drqn.yaml")
    movement_cfg_training = NNConfigMovement("./pretrained_weights/dqn/nn_config.yaml")
    env = PuckRealMultiworldFinal(movement_cfg_training, sim_cfg, FRONTEND_ACTIVE, training_goal=True)
    # pre_train_steps = 10000 #How many steps of random actions before training begins.
    # create lists to contain total rewards and steps per episode
    # load_network = False
    max_time = 390
    # extension = "_acc_reward_deactivated/"
    # if movement_cfg.use_angle_reward or movement_cfg.use_angle_reward == "True":
    #     extension = "_acc_reward_active/"
    movement_log_path = "./log/dqn/"
    movement_weight_path = "./weights_save/dqn/"
    # movement_cfg = MovementConfig2()

    angle_log_path = "./log/rnn/"
    angle_weight_path = "./weights_save/rnn/"
    # angle_cfg_training = NNConfigAngle(
    #     "../../tf_nn_sim/Config/NeuralNetwork/Angle/nn_config_training_single_layer.yaml")
    angle_cfg_training1 = NNConfigAngle("./pretrained_weights/rnn/nn_config.yaml")


    jList = []
    rList = []
    total_steps = 0
    # cfg = MediumConfig()
    # for angle_cfg_training in [angle_cfg_training1]:
    for angle_cfg_training in [AngleConfig1()]:
        CURRENT_BEST_ACCURACY = 0.0
        movement_cfg_predict = copy.deepcopy(movement_cfg_training)
        movement_cfg_predict.batch_size = 1
        movement_cfg_predict.keep_prob = 1.0

        angle_cfg_predict = copy.deepcopy(angle_cfg_training)
        angle_cfg_predict.batch_size = 1
        angle_cfg_predict.keep_prob = 1.0

        # learn_rate_range = movement_cfg.max_learning_rate - cfg.min_learning_rate
        percent = 0.0
        epsi_per = 0.0
        with tf.Graph().as_default():
            initializer_angle = tf.random_normal_initializer(stddev=angle_cfg_predict.init_scale)
            initializer_movement = tf.random_normal_initializer(stddev=movement_cfg_predict.init_scale)
            mov_gen_weight_path, mov_gen_log_path, load_movement_weights = create_nn_folders(movement_cfg_training,
                                                                                             movement_weight_path,
                                                                                             movement_log_path, False)
            movement_cfg_predict.weight_folder = movement_cfg_training.weight_folder
            movement_cfg_predict.log_folder = movement_cfg_training.log_folder
            angle_gen_weight_path, angle_gen_log_path, load_angle_weights = create_nn_folders(angle_cfg_training,
                                                                                              angle_weight_path,
                                                                                              angle_log_path)
            angle_cfg_predict.weight_folder = angle_cfg_training.weight_folder
            angle_cfg_predict.log_folder = angle_cfg_training.log_folder
            # with tf.name_scope("Train"):

            with tf.variable_scope("MovementModel", reuse=None, initializer=initializer_movement) as scope:
                predict_network_movement = DQN(config=movement_cfg_predict, namespace="DQN", is_training=False,
                                               log_path=movement_cfg_predict.log_folder,
                                               weight_path=movement_cfg_predict.weight_folder, variable_scope=scope,
                                               num_actions=env.num_actions,
                                               num_states=env.num_states)

                # with tf.name_scope("Train"):
            with tf.variable_scope("Angle_Model_Discrete", reuse=None, initializer=initializer_angle) as angle_scope:
                angle_training_network = GeneralRRNDiscreteModelMultitaskJointLoss(num_drones=sim_cfg.num_drones,
                                                                                   config=angle_cfg_training,
                                                                                   namespace="angle",
                                                                                   is_training=True,
                                                                                   log_path=angle_cfg_training.log_folder,
                                                                                   weight_path=angle_cfg_training.weight_folder,
                                                                                   variable_scope=angle_scope)
            with tf.variable_scope("Angle_Model_Discrete", reuse=True, initializer=initializer_angle) as angle_scope:
                angle_predict_network = GeneralRRNDiscreteModelMultitaskJointLoss(num_drones=sim_cfg.num_drones,
                                                                                  config=angle_cfg_predict,
                                                                                  namespace="angle",
                                                                                  is_training=False,
                                                                                  log_path=angle_cfg_predict.log_folder,
                                                                                  weight_path=angle_cfg_predict.weight_folder,
                                                                                  variable_scope=angle_scope)
                angle_predict_network.replay_buffer = angle_training_network.replay_buffer

            start_time = time.time()
            learn_rate_range = angle_cfg_training.max_learning_rate - angle_cfg_training.min_learning_rate
            current_learn_rate = movement_cfg_training.max_learning_rate
            epsilon_range = movement_cfg_training.epsilon_start - movement_cfg_training.epsilon_end
            training_worker_angle = TrainingWorker()
            keep = 0.0
            # sv = tf.train.Supervisor(logdir=angle_path)
            init = tf.global_variables_initializer()
            max_reward = -999999999.0
            max_accuracy_angle = 0.0
            max_accuracy_orientation = 0.0
            with tf.Session() as sess:
                sess.run(init)
                if load_movement_weights:
                    print('Loading Movement Model...')
                    predict_network_movement.load_weight(sess)
                if load_angle_weights:
                    print('Loading Angle Model...')
                    angle_predict_network.load_weight(sess)
                # for i in range(env.num_episodes):
                angle_training_network.init_logger(sess)
                angle_predict_network.init_logger(sess)
                episode_count = 0
                rewards_avg = collections.deque(maxlen=5)
                angle_accuracy_avg = collections.deque(maxlen=5)
                angle_accuracy_avg_episode = collections.deque(maxlen=5)
                orientation_accuracy_avg_episode = collections.deque(maxlen=5)
                while True:

                    s = env.reset()
                    d = False
                    step_count = 0
                    rrn_states_obstacle_angle = []
                    rrn_states_goal_angle = []
                    reward_per_episode = []
                    accuracy_per_episode = []
                    accuracy_per_episode_orientation = []
                    env.speed_limit = np.random.uniform(0.2, movement_cfg_training.speed_limit)
                    for d in range(env.num_drones):
                        rrn_states_obstacle_angle.append(angle_predict_network.gen_init_state(sess, 1))
                        rrn_states_goal_angle.append(angle_predict_network.gen_init_state(sess, 1))

                    for step_count in range(int(env.max_epLength * env.update_rate)):
                        start_frame = time.time()
                        current_learn_rate = movement_cfg_training.max_learning_rate - learn_rate_range * percent
                        movement_cfg_training.epsilon = 0.0

                        # if step_count == 0:
                        #     s = append_angle_state(s?, predict_angle)
                        # s1, actions, r, done, debug_data
                        s1, actions, r, crash, debug_d, goal_collect = trigger_movement_nn(sess, env, s,
                                                                                           predict_network_movement,
                                                                                           movement_cfg_training)
                        # return rrn_states_goal_angle, rrn_states_obstacle_angle, rrn_states_goal_orientation, rrn_states_obstacle_orientation, predict_angle_obs, predict_angle_goal, error
                        # def trigger_angle_nn(sess, predict_network, env, rrn_states_goal_angle,
                        #                      rrn_states_goal_orientation,
                        #                      rrn_states_obstacle_angle, rrn_states_obstacle_orientation, cfg,
                        #                      episode_count):
                        rrn_states_goal_angle, rrn_states_obstacle_angle, predict_angle_obs, predict_angle_goal, error, error_orientation = trigger_angle_nn(
                            sess,
                            angle_predict_network,
                            env,
                            rrn_states_goal_angle, rrn_states_obstacle_angle,
                            angle_cfg_training, episode_count)

                        # s1, r = use_predict_angle(env, s1, r, predict_angle_obs, predict_angle_goal, error,
                        #                           movement_cfg_predict.use_angle_reward)
                        accuracy_per_episode.append(1.0 - (sum(error) / len(error)))
                        if step_count > angle_cfg_predict.min_history:
                            accuracy_per_episode_orientation.append(
                                1.0 - (sum(error_orientation) / len(error_orientation)))
                        reward_per_episode.append(sum(r) / len(r))
                        # if episode_count > movement_cfg_training.pre_train_episodes and not training_worker_angle.is_training:
                        if episode_count > 2 and not training_worker_angle.is_training:
                            training_worker_angle.start_training(sess, env, angle_training_network,
                                                                 angle_predict_network, angle_cfg_training)
                        if training_worker_angle.is_training:
                            training_worker_angle.resume()
                        reward_per_episode.extend(r)
                        s = copy.copy(s1)
                        env.render(debug_d)
                        total_steps += 1
                        if FRONTEND_ACTIVE and env.frontend.real_time_active():
                            try:
                                passed_time = time.time() - start_frame
                                time.sleep((1.0 / env.update_rate) - passed_time)
                            except IOError:
                                print "ERROR: passed_time {0}".format(passed_time)
                        if True in crash:
                            break

                        if total_steps > 2:
                            for d_id in range(env.num_drones):
                                if goal_collect[d_id] and not env.training_goal:
                                    rrn_states_goal_angle[d_id] = angle_predict_network.gen_init_state(sess, 1)
                    r_all_episode = float(sum(reward_per_episode)) / len(reward_per_episode)
                    acc_all_episode = float(sum(accuracy_per_episode)) / len(accuracy_per_episode)
                    acc_all_episode_orientation = np.mean(accuracy_per_episode_orientation)
                    rewards_avg.append(r_all_episode)
                    angle_accuracy_avg.append(training_worker_angle.accuracy)
                    angle_accuracy_avg_episode.append(acc_all_episode)
                    orientation_accuracy_avg_episode.append(acc_all_episode_orientation)
                    episode_count += 1
                    percent = (time.time() - start_time) / (max_time * 60)
                    learn_rate_per = (time.time() - start_time) / (angle_cfg_training.learning_rate_time * 60)
                    learn_rate_per = 1.0 if learn_rate_per > 1.0 else learn_rate_per
                    current_learn_rate = angle_cfg_training.max_learning_rate - learn_rate_range * learn_rate_per
                    print "percent done: {0}% learn rate {1} epsilon {2} reward {3} accuracy_valid {4} acc_epi {5} acc_orientation {6}".format(
                        percent * 100,
                        current_learn_rate,
                        movement_cfg_training.epsilon,
                        r_all_episode,
                        np.mean(angle_accuracy_avg),
                        np.mean(angle_accuracy_avg_episode), np.mean(orientation_accuracy_avg_episode))
                    if (np.mean(angle_accuracy_avg_episode) > max_accuracy_angle or np.mean(
                            orientation_accuracy_avg_episode) > max_accuracy_orientation) and len(
                            angle_accuracy_avg_episode) > 4:
                        if np.mean(angle_accuracy_avg_episode) > max_accuracy_angle:
                            max_accuracy_angle = np.mean(angle_accuracy_avg_episode)
                        if np.mean(orientation_accuracy_avg_episode) > max_accuracy_orientation:
                            max_accuracy_orientation = np.mean(orientation_accuracy_avg_episode)
                        print "save angle weights angle {} orientation {}".format(max_accuracy_angle, max_accuracy_orientation)
                        angle_training_network.save_weights_to_folder(sess, episode_count, angle_gen_weight_path)
                    if percent >= 1.0:
                        break
                training_worker_angle.stop_training()
