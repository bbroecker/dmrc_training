import copy
import os
import threading
import time

import numpy as np
import tensorflow as tf
import collections

# from dbmrc_training.Config.NNConfigAngle import NNConfigAngle
# from tf_nn_sim.Config.NNConfigMovement import NNConfigMovement
# from tf_nn_sim.Config.SimConfig import SimulationConfig
# from tf_nn_sim.Environments.PuckRealMulti.Puckworld_Environment import PuckRealMultiworld
# from tf_nn_sim.Environments.PuckRealMultiFinal.Puckworld_Environment import PuckRealMultiworldFinal
# from tf_nn_sim.networks.dqn import DQN
# from tf_nn_sim.networks.rrn_cnn import GeneralRRNDiscreteModel
# import collections
#
# from tf_nn_sim.networks.rrn_cnn_multitask_join import GeneralRRNDiscreteModelMultitaskJointLoss
# from tf_nn_sim.networks.rrn_dqn import DQN_RNN
# from tf_nn_sim.v2.helper import translate, create_nn_folders
# from tf_nn_sim.v2.logger.logger import DataLogger
# from tf_nn_sim.v2.network_wrapper import AngleNetworkWrapper
# from tf_nn_sim.v2.particle_filter.particle_filter_nn import ParticleFilterNN, ParticleFilter2_5D_Cfg
from Config.NNConfigAngle import NNConfigAngle
from Config.NNConfigMovement import NNConfigMovement
from Config.SimConfig import SimulationConfig
from Environments.PuckRealMultiFinal.Puckworld_Environment import PuckRealMultiworldFinal
from networks.dqn import DQN
from networks.rrn_cnn_multitask_join import GeneralRRNDiscreteModelMultitaskJointLoss
from v2.helper import translate, create_nn_folders
from v2.logger.logger import DataLogger
from v2.network_wrapper import AngleNetworkWrapper
from v2.particle_filter.particle_filter_nn import ParticleFilter2_5D_Cfg

FRONTEND_ACTIVE = True


def discrete_to_angle(discrete_angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    return step_size * discrete_angle - np.pi


def angle_to_discrete(angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    index = int((angle + np.pi) / step_size)
    return index


def trigger_angle_nn(sess, env, angle_wrapper, angle_cfg, logger, reset=False):
    env.predict_obstacle_angle = [0.0] * env.num_drones
    env.predict_goal_angle = [0.0] * env.num_drones
    predict_angle_obs = [0.0] * env.num_drones
    predict_angle_goal = [0.0] * env.num_drones
    env.predict_obstacle_orientation = {}
    env.particle_obs_predict = {}
    predict_orientation_obs = [0.0] * env.num_drones
    predict_orientation_goal = [0.0] * env.num_drones
    error = [0.0] * env.num_drones
    error_particle = [0.0] * env.num_drones
    for d_id in range(env.num_drones):
        # c_id = env.closes_drone(d_id)
        vx = env.pvx[d_id]
        vy = env.pvy[d_id]
        o_vx = 0.0
        o_vy = 0.0
        g_distance = env.get_goal_distance(d_id, mean=0.0, variance=0.0312979988458)
        # g_distance = env.get_goal_distance(d_id)
        o_distance = angle_cfg_predict.max_sensor_distance

        goal_angle = env.get_goal_angle(d_id)
        angle_wrapper[d_id].update_goal_particle_filter(vx, vy, 0.0, 0.0, g_distance, 1.0 / env.update_rate)
        goal_x, goal_y, goal_yaw = angle_wrapper[d_id].get_goal_estimate_pose()
        predict_angle_goal[d_id] = np.arctan2(goal_y, goal_x)

        for o_id in range(env.num_drones):
            if d_id == o_id:
                continue
            o_vx = env.pvx[o_id]
            o_vy = env.pvy[o_id]
            o_distance = env.get_distance_to(d_id, o_id, mean=0.0, variance=0.0312979988458)
            # o_distance = env.get_distance_to(d_id, o_id)
            # print "o_distance ", o_distance, d_id, o_id
            angle_wrapper[d_id].append_obs_state(o_id, vx, vy, o_vx, o_vy, o_distance, 1.0 / env.update_rate)
        angle_wrapper[d_id].update_obs_particle_filter()
        obs_x, obs_y, predict_orientation_obs[d_id] = angle_wrapper[d_id].get_obs_estimate_pose()
        obs_angle_network = angle_wrapper[d_id].get_obs_network_angle_prediction()
        goal_angle_network = angle_wrapper[d_id].get_goal_network_angle_prediction()
        predict_angle_obs[d_id] = np.arctan2(obs_y, obs_x)
        obs_angle = env.get_obstacle_angle(d_id)
        # angle = int(translate(angle, -1.0, 1.0, 0, cfg.output_size - 0.001))
        # print "goal {} predict {}".format(angle_to_discrete(goal_angle, cfg), predict_angle_goal[d_id])
        # er1 = 0.0 if abs(angle_to_discrete(goal_angle, cfg) - predict_angle_goal[d_id]) < 2 else 1.0
        er1 = abs(((abs((goal_angle - goal_angle_network) / np.pi) + 1.0) % (2.0)) - 1.0)
        er2 = abs(((abs((obs_angle - obs_angle_network) / np.pi) + 1.0) % (2.0)) - 1.0)

        error[d_id] = (abs(er1) + abs(er2)) / 2.0
        # error[d_id] = abs(er2)

        if step_count > angle_cfg_predict.min_history:
            logger.insert_data(d_id, "error_nn_goal", er1)
            logger.insert_data(d_id, "error_nn_obs", er2)
        env.predict_obstacle_orientation[d_id] = predict_orientation_obs[d_id]

        if reset:
            pass
            # particles_obs[d_id].reset(o_x, o_y, discrete_to_angle(predict_orientation_obs[d_id][0], cfg))

        x, y = angle_wrapper[d_id].global_pos_particle_obs(env.ppx[d_id], env.ppy[d_id], env.orientation[d_id])
        # env.particle_obs_predict[d_id] = [x, y]
        x, y = angle_wrapper[d_id].global_pos_particle_goal(env.ppx[d_id], env.ppy[d_id], env.orientation[d_id])
        env.particle_goal_predict[d_id] = [x, y]
        # print (obs_angle - particles_obs[d_id].get_angle())/np.pi
        er1 = abs(((abs((goal_angle - predict_angle_goal[d_id]) / np.pi) + 1.0) % (2.0)) - 1.0)
        er2 = abs(((abs((obs_angle - predict_angle_obs[d_id]) / np.pi) + 1.0) % (2.0)) - 1.0)
        error_particle[d_id] = (abs(er1) + abs(er2)) / 2.0
        # error_particle[d_id] = abs(er2)
        if step_count > angle_cfg_predict.min_history:
            logger.insert_data(d_id, "error_particle_goal", er1)
            logger.insert_data(d_id, "error_particle_obs", er2)
        env.set_goal_partices(d_id, angle_wrapper[d_id].get_goal_particles())
        # env.set_obs_partices(d_id, angle_wrapper[d_id].get_obs_particles())

        # env.predict_obstacle_angle[d_id] = predict_angle_obs[d_id]
        # env.predict_obstacle_angle[d_id] = predict_angle_obs[d_id]
        env.predict_obstacle_angle[d_id] = obs_angle_network
        # env.predict_obstacle_angle[d_id] = obs_angle
        # env.predict_goal_angle[d_id] = predict_angle_goal[d_id]
        env.predict_goal_angle[d_id] = goal_angle_network
        # env.predict_goal_angle[d_id] = goal_angle

    return predict_angle_obs, predict_angle_goal


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
        if np.random.rand(1) < cfg.epsilon:
            actions[d_id] = np.random.randint(0, env.num_actions)
            # actions[d_id] = 2
            # a = 5
        else:
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


def use_predict_angle(env, s1, r, angle_network_wrapper):
    for d_id in range(env.num_drones):
        g_dx, g_dy, _ = angle_network_wrapper[d_id].get_goal_estimate_pose()
        distance = np.sqrt(g_dx ** 2 + g_dy ** 2)
        g_dx /= distance
        g_dx *= distance + 0.40
        g_dy /= distance
        g_dy *= distance + 0.40
        g_dx = translate(g_dx, -movement_cfg_predict.max_sensor_distance, movement_cfg_predict.max_sensor_distance,
                         angle_cfg_predict.min_x, angle_cfg_predict.max_y)
        g_dy = translate(g_dy, -movement_cfg_predict.max_sensor_distance, movement_cfg_predict.max_sensor_distance,
                         angle_cfg_predict.min_y, angle_cfg_predict.max_y)

        o_dx, o_dy, _ = angle_network_wrapper[d_id].get_obs_estimate_pose()
        # print o_dx, o_dy
        o_dx = translate(o_dx, -movement_cfg_predict.max_sensor_distance, movement_cfg_predict.max_sensor_distance,
                         angle_cfg_predict.min_x, angle_cfg_predict.max_x)
        o_dy = translate(o_dy, -movement_cfg_predict.max_sensor_distance, movement_cfg_predict.max_sensor_distance,
                         angle_cfg_predict.min_y, angle_cfg_predict.max_y)

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
    max_learning_rate = 0.0004
    # max_learning_rate = 0.001
    min_learning_rate = 0.0004
    # min_learning_rate = 0.001
    learning_rate_time = 120.0
    max_grad_norm = 30
    num_layers = 2
    state_count = 6
    # state_count = 3
    trace_length = 50
    min_history = 30
    buffer_size = 1
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

    # goal_samples_size: 30
    # spawn_new_goal_samples: 0.1
    # goal_resample_x_limit: 0.05
    # goal_resample_y_limit: 0.05
    # goal_resample_yaw_limit: 0.1
    # goal_respawn_yaw_limit: 0.1
    # goal_respawn_y_limit: 0.01
    # goal_respawn_x_limit: 0.01
    # goal_resample_prob: 0.95
    #
    # obs_samples_size: 70
    # spawn_new_obs_samples: 0.1
    # #in meter
    # obs_resample_x_limit: 0.3
    # obs_resample_y_limit: 0.3
    # #percent of pi
    # obs_resample_yaw_limit: 0.1
    # obs_respawn_yaw_limit: 0.1
    # #in meter
    # obs_respawn_y_limit: 0.1
    # obs_respawn_x_limit: 0.1
    # obs_resample_prob: 0.70


class ObstacleCfg(ParticleFilter2_5D_Cfg):
    def __init__(self):
        super(ObstacleCfg, self).__init__()
        self.sample_size = 30
        # self.sample_size = 600
        self.x_limit = [-1.0, 1.0]
        self.resample_x_limit = [-0.01, 0.01]
        self.y_limit = [-1.0, 1.0]
        self.resample_y_limit = [-0.01, 0.01]
        self.yaw_limit = [-np.pi, np.pi]
        self.resample_yaw_limit = [-np.pi * 0.1, np.pi * 0.1]
        self.respawn_yaw_limit = [-np.pi * 0.1, np.pi * 0.1]
        self.respawn_y_limit = [-0.01, 0.01]
        self.respawn_x_limit = [-0.01, 0.01]
        self.new_spawn_samples = 0.1
        # self.new_spawn_samples = 0.0
        self.resample_prob = 0.9
        # self.resample_prob = 0.0


class GoalCfg(ParticleFilter2_5D_Cfg):
    def __init__(self):
        super(GoalCfg, self).__init__()
        # self.sample_size = 35
        self.sample_size = 30
        self.x_limit = [-1.0, 1.0]
        self.resample_x_limit = [-0.1, 0.1]
        # self.resample_x_limit = [-0.0, 0.0]
        self.y_limit = [-1.0, 1.0]
        self.resample_y_limit = [-0.1, 0.1]
        # self.resample_y_limit = [-0.0, 0.0]
        self.yaw_limit = [-np.pi, np.pi]
        self.resample_yaw_limit = [-np.pi * 0.01, np.pi * 0.01]
        self.respawn_yaw_limit = [-np.pi * 0.1, np.pi * 0.1]
        # self.respawn_y_limit = [-0.01, 0.01]
        self.respawn_y_limit = [-0.01, 0.01]
        # self.respawn_x_limit = [-0.01, 0.01]
        self.respawn_x_limit = [-0.01, 0.01]
        self.new_spawn_samples = 0.1
        # self.new_spawn_samples = 0.0
        self.resample_prob = 0.95


class TrainingWorker(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.is_training = False

    def start_training(self, sess, env, training_network, cfg):
        self.is_training = True
        self.sess = sess
        self.env = env
        self.training_network = training_network
        self.cfg = cfg
        self.pause = False
        self.start()

    def resume(self):
        self.pause = False

    def run(self):
        steps = 0
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
            self.pause = True

    def stop_training(self):
        self.is_training = False


if __name__ == '__main__':
    print os.getcwd()
    # env = Puckworld()
    # env = PuckNormalMultiworld()
    sim_cfg = SimulationConfig("../Config/Simulation/sim_config.yaml")
    sim_cfg.goal_update_distance = 0.2
    sim_cfg.num_drones = 2
    # movement_cfg = NNConfigMovement("/home/broecker/src/cf_nn_weights/dqn/drqn_1/nn_config_drqn.yaml")
    movement_cfg = NNConfigMovement("./pretrained_weights/dqn/nn_config.yaml")
    movement_cfg.update_rate = 8.0
    env = PuckRealMultiworldFinal(movement_cfg, sim_cfg, FRONTEND_ACTIVE, training_goal=False)
    # pre_train_steps = 10000 #How many steps of random actions before training begins.
    # create lists to contain total rewards and steps per episode
    # load_network = False
    max_time = 2900
    # extension = "_acc_reward_deactivated/"
    # if movement_cfg.use_angle_reward or movement_cfg.use_angle_reward == "True":
    #     extension = "_acc_reward_active/"
    movement_log_path = "./log/dqn/particle/batch/"
    movement_weight_path = "./weights_save/dqn/particle/batch/"
    # movement_cfg = MovementConfig2()

    angle_log_path = "./log/rnn/"
    angle_weight_path = "./weights_save/rnn/"
    # angle_cfg_training = NNConfigAngle(
    #     "../../tf_nn_sim/Config/NeuralNetwork/Angle/nn_config_training_single_layer.yaml")
    angle_cfg_training = NNConfigAngle("./pretrained_weights/rnn/nn_config.yaml")
    # "/home/broecker/src/cf_nn_weights/v2/angle_discrete/orientation_joint/rnn_1/nn_config.yaml"
    # angle_cfg_training = AngleConfig1()
    # angle_cfg_training.weight_folder = "/home/broecker/src/ros/swarm_avoid_ws/src/tf_nn_sim/src/tf_nn_sim/v2/weights_save/rnn_dynamic/angle_discrete/batch/lr0.001_rnn_tanh512_tanh512_dense_batch30_trace_length50_update_length10_forget_bias0.0_init_value0.04419_keep_prob0.8_buffer_size1_RMSPropOptimizer_2"
    # angle_cfg_training.min_learning_rate = 0.001
    # angle_cfg_training.max_learning_rate = 0.001
    angle_cfg_predict = copy.copy(angle_cfg_training)
    angle_cfg_predict.batch_size = 1
    angle_cfg_predict.keep_prob = 1.0
    jList = []
    rList = []
    total_steps = 0
    env.speed_limit = 0.4
    # cfg = MediumConfig()
    for movement_cfg_training in [movement_cfg]:
        CURRENT_BEST_ACCURACY = 0.0
        movement_cfg_predict = copy.copy(movement_cfg_training)
        movement_cfg_predict.batch_size = 1
        movement_cfg_predict.keep_prob = 1.0

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
                                                                                              angle_log_path, False)
            angle_cfg_predict.weight_folder = angle_cfg_training.weight_folder
            angle_cfg_predict.log_folder = angle_cfg_training.log_folder
            # with tf.name_scope("Train"):

            with tf.variable_scope("MovementModel", reuse=None, initializer=initializer_movement) as scope:
                predict_network_movement = DQN(config=movement_cfg_predict, namespace="DQN", is_training=False,
                                               log_path=movement_cfg_predict.log_folder,
                                               weight_path=movement_cfg_predict.weight_folder, variable_scope=scope,
                                               num_actions=env.num_actions,
                                               num_states=env.num_states)

            with tf.variable_scope("Angle_Model_Discrete", reuse=None, initializer=initializer_angle) as angle_scope:
                angle_predict_network = GeneralRRNDiscreteModelMultitaskJointLoss(num_drones=sim_cfg.num_drones,
                                                                                  config=angle_cfg_predict,
                                                                                  namespace="angle",
                                                                                  is_training=False,
                                                                                  log_path=angle_cfg_predict.log_folder,
                                                                                  weight_path=angle_cfg_predict.weight_folder,
                                                                                  variable_scope=angle_scope)

            start_time = time.time()

            epsilon_range = movement_cfg_training.epsilon_start - movement_cfg_training.epsilon_end
            keep = 0.0
            # sv = tf.train.Supervisor(logdir=angle_path)
            init = tf.global_variables_initializer()
            max_reward = -999999999.0
            max_accuracy = 0.0
            angle_network_wrapper = [0] * sim_cfg.num_drones
            with tf.Session() as sess:
                sess.run(init)
                if load_movement_weights:
                    print('Loading Movement Model...')
                    predict_network_movement.load_weight(sess)
                if load_angle_weights:
                    print('Loading Angle Model...')
                    angle_predict_network.load_weight(sess)
                episode_count = 0
                rewards_avg = collections.deque(maxlen=5)
                angle_accuracy_avg = collections.deque(maxlen=5)
                best_reward = movement_cfg_training.best_score
                reward_log_count = 0
                while True:

                    s = env.reset()
                    d = False
                    step_count = 0
                    rAll = 0.0
                    rrn_states_dqn = []
                    reward_per_episode = []
                    accuracy_per_episode = []
                    accuracy_per_episode_particle = []
                    logger = DataLogger()
                    particles_goal = []
                    particles_obs = []
                    rewards = []
                    reward_avg = collections.deque(maxlen=6)
                    angle_network_wrapper = []
                    goals_collected = 0
                    for d in range(env.num_drones):
                        angle_network_wrapper.append(
                            AngleNetworkWrapper(sess, angle_predict_network, 3, ObstacleCfg(), GoalCfg()))

                    for step_count in range(int(env.max_epLength * env.update_rate)):
                        start_frame = time.time()
                        movement_cfg_training.epsilon = 0.0

                        # if step_count == 0:
                        #     s = append_angle_state(s?, predict_angle)
                        # s1, actions, r, done, debug_data
                        s1, actions, r, crash, debug_d, goal_collect = trigger_movement_nn(sess, env, s,
                                                                                           predict_network_movement,
                                                                                           movement_cfg_training)
                        goals_collected += sum(goal_collect)

                        predict_angle_obs, predict_angle_goal = trigger_angle_nn(
                            sess,
                            env,
                            angle_network_wrapper, angle_cfg_predict,
                            logger, step_count == 0)

                        s1, r = use_predict_angle(env, s1, r, angle_network_wrapper)
                        rewards.extend(r)

                        # logger.insert_data("reward", r)
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
                            print "crash"
                            break

                        if total_steps > 2:
                            for d_id in range(env.num_drones):
                                if goal_collect[d_id]:
                                    # rrn_states_dqn[d_id] = predict_network_movement.gen_init_state(sess, 1)
                                    angle_network_wrapper[d_id].reset_goal()
                    rAll = float(goals_collected) / env.num_drones
                    reward_avg.append(rAll / step_count)
                    reward_log_count += 1

                    episode_count += 1
                    percent = (time.time() - start_time) / (max_time * 60)
                    logger.print_data()

                    # logger.draw_graph(0, ["error_nn_goal", "error_particle_goal"], ['r', 'b'])
                    print "percent done: {0}%  epsilon {1} reward {2} update_rate {3}".format(
                        percent * 100,
                        movement_cfg_training.epsilon,
                        rAll / step_count, env.update_rate)
                    if percent >= 1.0:
                        break

