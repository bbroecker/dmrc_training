import rospy
import tf
from geometry_msgs.msg import PoseStamped

import numpy as np

from v2.particle_filter.particle_filter_nn import ParticleFilterNN, rotate_vel

MAX_DISTANCE = 9999999999.9


def discrete_to_angle(discrete_angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    return step_size * discrete_angle - np.pi


def angle_to_discrete(angle, cfg):
    range_size = 2 * np.pi
    step_size = range_size / float(cfg.output_size - 1.0)
    index = int(round((angle + np.pi) / step_size))
    return index

def quaternon_from_yaw(yaw):
    quaternion = tf.transformations.quaternion_from_euler(0, 0, yaw)
    return quaternion[0], quaternion[1], quaternion[2], quaternion[3]

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


class ObstacleStateEstimator:
    def __init__(self, sess, network, particle_cfg, obs_id):
        self.distance = MAX_DISTANCE
        self.sess = sess
        self.obs_id = obs_id
        self.network = network
        self.nn_x = 0
        self.nn_y = 0
        self.nn_yaw = 0
        self.rnn_state = network.gen_init_state(sess, 1)
        self.network_angle_prediction = None
        self.network_orientation_prediction = None
        self.particle_filter = ParticleFilterNN(particle_cfg)

    def reset(self):
        self.rnn_state = self.network.gen_init_state(self.sess, 1)
        self.particle_filter.reset()

    def translate_state(self, state):
        min_output = self.network.config.min_output
        max_output = self.network.config.max_output
        distance_translate = self.translate(state.distance, 0, self.angle_cfg.max_sensor_distance, min_output,
                                            max_output)
        vx = self.translate(state.vx, -self.angle_cfg.max_velocity, self.angle_cfg.max_velocity, min_output, max_output)
        vy = self.translate(state.vy, -self.angle_cfg.max_velocity, self.angle_cfg.max_velocity, min_output, max_output)
        o_vx = self.translate(state.o_vx, -self.angle_cfg.max_velocity, self.angle_cfg.max_velocity, min_output,
                              max_output)
        o_vy = self.translate(state.o_vy, -self.angle_cfg.max_velocity, self.angle_cfg.max_velocity, min_output,
                              max_output)

        return [vx, vy, o_vx, o_vy, distance_translate, state.dt]

    def update(self, s):
        assert isinstance(s, State)
        self.distance = s.distance
        state = self.network.translate_state(s.vx, s.vy, s.o_vx, s.o_vy, s.distance, s.dt)
        # print "translate {}".format(s.o_id), state
        # print state
        predict_angle, predict_orientation, self.rnn_state = self.network.predict_angle_orientation(self.sess, state,
                                                                                                    self.rnn_state)
        # print predict_angle
        self.network_angle_prediction = discrete_to_angle(predict_angle[0], self.network.config)
        self.network_orientation_prediction = discrete_to_angle(predict_orientation[0], self.network.config)

        g_x = s.distance * np.cos(self.network_angle_prediction)
        g_y = s.distance * np.sin(self.network_angle_prediction)
        self.nn_x = g_x
        self.nn_y = g_y
        self.nn_yaw = self.network_orientation_prediction
        # print "update obs ", g_x, g_y, self.network_angle_prediction, self.network_orientation_prediction

        # if reset:
        #     pass
        # particles_obs[d_id].reset(o_x, o_y, discrete_to_angle(predict_orientation_obs[d_id][0], cfg))
        # particles_goal[d_id].reset(g_x, g_y, discrete_to_angle(predict_orientation_obs[d_id][0], cfg))
        # print vx, vy, 0.0, 0.0, 1.0 / env.update_rate, g_distance, g_x, g_y
        # print s.vx, s.vy, s.o_vx, s.o_vy, s.dt, s.distance, g_x, g_y, self.network_orientation_prediction
        self.particle_filter.update_samples(s.vx, s.vy, s.o_vx, s.o_vy, s.dt, s.distance, g_x, g_y,
                                            self.network_orientation_prediction)

    def local_pose(self):
        return self.particle_filter.get_local_pose()


class State:
    def __init__(self, o_id, vx, vy, o_vx, o_vy, dt, distance):
        self.o_id = o_id
        self.vx = vx
        self.vy = vy
        self.o_vx = o_vx
        self.o_vy = o_vy
        self.dt = dt
        self.distance = distance


class AngleNetworkWrapper:
    def __init__(self, sess, angle_predict_network, k_neigthbours, particle_obs_cfg, particle_goal_cfg, frame_id = None):

        self.angle_predict_network = angle_predict_network
        self.state_obs = []
        self.particle_obs_cfg = particle_obs_cfg
        self.particle_goal_cfg = particle_goal_cfg
        self.particle_goal_active = particle_goal_cfg.sample_size > 0
        self.particle_obs_active = particle_obs_cfg.sample_size > 0
        self.nn_goal_active = particle_goal_cfg.new_spawn_samples > 0
        self.nn_obs_active = particle_obs_cfg.new_spawn_samples > 0
        self.sess = sess
        self.k_neigthbours = k_neigthbours
        self.obs_state_estimator = []
        self.rnn_goal_state = angle_predict_network.gen_init_state(sess, 1)
        self.particle_filter_goal = ParticleFilterNN(particle_goal_cfg)
        self.goal_angle_estimation = None
        self.goal_orientation_estimation = None

        self.nn_goal_x = 0
        self.nn_goal_y = 0
        self.nn_goal_yaw = 0

        self.nn_obs_x = 0
        self.nn_obs_y = 0
        self.nn_obs_yaw = 0

        self.frame_id = frame_id
        if frame_id is not None:
            self.predict_goal_network_pub = rospy.Publisher("predict_goal_nn", PoseStamped, queue_size=1)
            self.predict_obs_network_pub = rospy.Publisher("predict_obs_nn", PoseStamped, queue_size=1)
            self.predict_goal_particle_pub = rospy.Publisher("predict_goal_particle", PoseStamped, queue_size=1)
            self.predict_obs_particle_pub = rospy.Publisher("predict_obs_particle", PoseStamped, queue_size=1)

    def reset_obs(self):
        # print "reset_obs"
        for estimator in self.obs_state_estimator:
            estimator.reset()

    def reset_goal(self):
        # print "reset_goal"
        self.particle_filter_goal.reset()
        self.rnn_goal_state = self.angle_predict_network.gen_init_state(self.sess, 1)

    def append_obs_state(self, o_id, vx, vy, o_vx, o_vy, distance, dt):
        # if o_id == 0:
        #     print "real 0: {} {} {} {}".format(vx, vy, o_vx, vy, distance)
        # print "append_obs_state", distance
        self.state_obs.append(State(o_id, vx, vy, o_vx, o_vy, dt, distance))

    def place_holder_state(self):
        return State(-1, 0.0, 0.0, 0.0, 0.0, 1.0 / self.angle_predict_network.config.update_rate,
                     self.angle_predict_network.config.max_sensor_distance)

    def update_obs_particle_filter(self):
        if len(self.state_obs) <= 0:
            self.state_obs.append(self.place_holder_state())
        self.state_obs.sort(key=lambda x: x.distance, reverse=False)
        self.state_obs = self.state_obs[0: self.k_neigthbours]
        relevant_list = [self.state_obs[k].o_id for k in range(min(self.k_neigthbours, len(self.state_obs)))]
        new_estimators = []
        for e in self.obs_state_estimator:
            if e.obs_id in relevant_list:
                new_estimators.append(e)
                relevant_list.remove(e.obs_id)

        for r_id in relevant_list:
            new_estimators.append(self.create_estimator(r_id))
        self.state_obs.sort(key=lambda x: x.o_id, reverse=False)
        new_estimators.sort(key=lambda x: x.obs_id, reverse=False)

        for idx, s in enumerate(self.state_obs):
            # if s.o_id == 0:
            #     print " real 1: {} {} {} {}".format(s.vx, s.vy, s.o_vx, s.vy)
            new_estimators[idx].update(s)
        new_estimators.sort(key=lambda x: x.distance, reverse=False)

        self.obs_state_estimator = new_estimators
        self.state_obs = []

        self.nn_obs_x = self.obs_state_estimator[0].nn_x
        self.nn_obs_y = self.obs_state_estimator[0].nn_y
        self.nn_obs_yaw = self.obs_state_estimator[0].nn_yaw

        if self.frame_id is not None:
            x, y, yaw = self.get_obs_estimate_pose()
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quaternon_from_yaw(yaw)
            self.predict_obs_particle_pub.publish(pose)

            g_x = s.distance * np.cos(self.obs_state_estimator[0].network_angle_prediction)
            g_y = s.distance * np.sin(self.obs_state_estimator[0].network_angle_prediction)
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = g_x
            pose.pose.position.y = g_y
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quaternon_from_yaw(self.obs_state_estimator[0].network_orientation_prediction)
            self.predict_obs_network_pub.publish(pose)


    def create_estimator(self, obs_id):
        return ObstacleStateEstimator(self.sess, self.angle_predict_network, self.particle_obs_cfg, obs_id)



    def update_goal_particle_filter(self, vx, vy, o_vx, o_vy, distance, dt):
        state = self.angle_predict_network.translate_state(vx, vy, o_vx, o_vy, distance, dt)
        predict_angle, predict_orientation, self.rnn_goal_state = self.angle_predict_network.predict_angle_orientation(
            self.sess, state, self.rnn_goal_state)

        self.goal_angle_estimation = discrete_to_angle(predict_angle[0], self.angle_predict_network.config)
        self.goal_orientation_estimation = discrete_to_angle(predict_orientation[0], self.angle_predict_network.config)

        self.nn_goal_yaw = self.goal_orientation_estimation
        self.nn_goal_x = distance * np.cos(self.goal_angle_estimation)
        self.nn_goal_y = distance * np.sin(self.goal_angle_estimation)

        # if reset:
        #     pass
        # particles_obs[d_id].reset(o_x, o_y, discrete_to_angle(predict_orientation_obs[d_id][0], cfg))
        # particles_goal[d_id].reset(g_x, g_y, discrete_to_angle(predict_orientation_obs[d_id][0], cfg))
        # print vx, vy, 0.0, 0.0, 1.0 / env.update_rate, g_distance, g_x, g_y
        self.particle_filter_goal.update_samples(vx, vy, o_vx, o_vy, dt, distance, self.nn_goal_x, self.nn_goal_y,
                                                 self.goal_orientation_estimation)
        if self.frame_id is not None:
            x, y, yaw = self.get_goal_estimate_pose()
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quaternon_from_yaw(yaw)
            self.predict_goal_particle_pub.publish(pose)

            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = self.nn_goal_x
            pose.pose.position.y = self.nn_goal_y
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quaternon_from_yaw(self.goal_angle_estimation)
            self.predict_goal_network_pub.publish(pose)

    def get_obs_estimate_pose(self):
        if self.particle_goal_active:
            if len(self.obs_state_estimator) <= 0:
                distance = self.angle_predict_network.config.max_sensor_distance
                x = np.sin(0.0) * distance
                y = np.cos(0.0) * distance
                yaw = 0.0
            else:
                x, y, yaw = self.obs_state_estimator[0].particle_filter.local_pose()
            angle = np.arctan2(y, x)

            return x, y, yaw
        return self.nn_obs_x, self.nn_obs_y, self.nn_obs_y


    def get_goal_estimate_pose(self):
        if self.particle_goal_active:
            return self.particle_filter_goal.local_pose()
        else:
            return self.nn_goal_x, self.nn_goal_y, self.nn_goal_yaw

    def get_obs_network_angle_prediction(self):
        return self.obs_state_estimator[0].network_angle_prediction

    def get_goal_network_angle_prediction(self):
        return self.goal_angle_estimation

    def get_goal_network_orientation_prediction(self):
        return self.goal_orientation_estimation

    def global_pos_particle_obs(self, x, y, yaw):
        if self.particle_obs_active:
            return self.obs_state_estimator[0].particle_filter.global_pos(x, y, yaw)
        else:
            p_x, p_y = rotate_vel(self.nn_obs_x, self.nn_obs_y, yaw)
            return x + p_x, y + p_y

    def global_pose_particle_obs(self, x, y, yaw):
        if self.particle_obs_active:
            return self.obs_state_estimator[0].particle_filter.global_pose(x, y, yaw)
        else:
            # print "self.nn_obs_x {} , self.nn_obs_y {}".format(self.nn_obs_x, self.nn_obs_y)
            p_x, p_y = rotate_vel(self.nn_obs_x, self.nn_obs_y, yaw)
            return x + p_x, y + p_y, wrap_angle(self.nn_obs_yaw + yaw)

    def global_pos_particle_goal(self, x, y, yaw):
        if self.particle_goal_active:
            return self.particle_filter_goal.global_pos(x, y, yaw)
        else:
            p_x, p_y = rotate_vel(self.nn_goal_x, self.nn_goal_y, yaw)
            return x + p_x, y + p_y

    def global_pose_particle_goal(self, x, y, yaw):
        if self.particle_goal_active:
            return self.particle_filter_goal.global_pose(x, y, yaw)
        else:
            # return self.nn_goal_x + x, self.nn_goal_y + y, wrap_angle(self.nn_goal_yaw + yaw)
            p_x, p_y = rotate_vel(self.nn_goal_x, self.nn_goal_y, yaw)
            return x + p_x, y + p_y, wrap_angle(self.nn_goal_yaw + yaw)

    def get_goal_particles(self):
        return self.particle_filter_goal.samples

    def get_obs_particles(self):
        if len(self.obs_state_estimator) <= 0:
            return None
        return self.obs_state_estimator[0].particle_filter.samples
