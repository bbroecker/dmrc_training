import numpy as np

from Config.NNConfigMovement import NNConfigMovement
from Config.SimConfig import SimulationConfig
from Environments.Environment import Environment
from Environments.PuckRealMultiFinal.Frontend_Puck import PuckRealMultiFinalFrontend
import collections
CONFIG_FILE = "../Config/PuckRealMultiFinal2/puckworld.yaml"

# self.nn_config.min_state_out = -1.0
# self.nn_config.max_state_out = 1.0
# self.nn_config.min_reward = -1.0
# self.nn_config.max_reward = 1.0
# self.nn_config.min_state_out = -5.0
# self.nn_config.max_state_out = 5.0
# self.nn_config.min_reward = -1.0
# self.nn_config.max_reward = 1.0
HISTORY_LENGTH = 2000
TRUNC_TIME = 1.0
LEFT_PREF = 0.5
COLLISION_BUFFER = 0.20


class PuckRealMultiworldFinal(Environment):
    def __init__(self, nn_config, simulation_cfg, show=True, training_goal=False):
        assert isinstance(simulation_cfg, SimulationConfig)
        # assert isinstance(nn_config, NNConfigMovement)
        Environment.__init__(self, CONFIG_FILE)
        self.training_goal = training_goal
        self.max_velocity = None
        self.acceleration = None
        self.damping_per_sec = None
        self.brake_per_sec = None
        self.critical_radius = None
        self.wall_size_x = None
        self.wall_size_y = None
        self.actions = None
        self.x_actions_len = None
        self.num_actions = None
        self.drone_radius = None
        self.orientation = None
        self.ppx = None
        self.D = 0.0
        self.ppx_history = None
        self.prev_ppx = None
        self.ppy = None
        self.ppy_history = None
        self.prev_ppy = None
        self.pvx = None
        self.pvx_history = None
        self.pvy = None
        self.pvy_history = None
        self.tx = None
        self.ty = None
        self.tx_train = None
        self.ty_train = None
        self.t = None
        self.nn_config = nn_config
        self.update_rate = None
        self.predict_obstacle_angle = []
        self.predict_obstacle_orientation = {}
        self.particle_obs_predict = {}
        self.particle_goal_predict = {}
        self.predict_goal_angle = []
        self.predict_goal_angle_noise = []
        self.goal_update_time = None
        self.bad_speed = None
        self.total_meter_x = None
        self.total_meter_y = None
        self.area_pixel_x = None
        self.debug_pixel_x = None
        self.area_pixel_y = None
        self.debug_pixel_y = None
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.y_end = None
        self.span_x_start = None
        self.span_y_start = None
        self.span_x_end = None
        self.span_y_end = None
        self.num_drones = None
        self.max_sensor_distance = None
        self.push_distance = 0.1
        self.particles_goal = None
        self.particles_obs = None

        self.load_cfg(nn_config, simulation_cfg)
        # print self.max_sensor_distance
        # damping factor
        self.damping_per_tick = 1.0 - self.damping_per_sec / self.update_rate
        self.damping_per_tick = 0.0 if self.damping_per_tick < 0.0 else self.damping_per_tick
        self.brake_per_tick = 1.0 - self.brake_per_sec / self.update_rate
        self.brake_per_tick = 0.0 if self.brake_per_tick < 0.0 else self.brake_per_tick

        self.acceleration_per_tick = self.acceleration * 1.0 / self.update_rate
        self.update_goal_step = self.goal_update_steps
        self.show = show
        self.goal_count = 0
        self.crash_count = 0
        self.reset()
        self.num_states = len(self.get_state(0))
        if show:
            self.frontend = PuckRealMultiFinalFrontend(self)

    def set_update_rate(self, update_rate):
        self.update_rate = update_rate
        self.damping_per_tick = 1.0 - self.damping_per_sec / self.update_rate
        self.damping_per_tick = 0.0 if self.damping_per_tick < 0.0 else self.damping_per_tick
        self.acceleration_per_tick = self.acceleration * 1.0 / self.update_rate

    def reset(self):
        self.x_start = -self.wall_size_x / 2.0
        self.x_end = self.wall_size_x / 2.0
        self.y_start = -self.wall_size_y / 2.0
        self.y_end = self.wall_size_y / 2.0
        self.goal_count = 0
        self.crash_count = 0
        # self.ppx = np.random.random() * self.wall_size_x
        states = []
        # self.ppx_history = [collections.dequeue(maxlen=HISTORY_LENGTH)] * self.num_drones
        # self.ppy_history = [collections.dequeue(maxlen=HISTORY_LENGTH)] * self.num_drones
        # self.pvx_history = [collections.dequeue(maxlen=HISTORY_LENGTH)] * self.num_drones
        # self.pvy_history = [collections.dequeue(maxlen=HISTORY_LENGTH)] * self.num_drones
        self.particles_goal = [[] for i in range(self.num_drones)]
        self.particles_obs = [[] for i in range(self.num_drones)]
        for i in range(self.num_drones):
            self.goal_collected[i] = False
            self.crashed[i] = False
            self.goal_id[i] += 1
            self.obstacle_id[i] += 1
            self.ppx[i] = np.random.uniform(self.span_x_start, self.span_x_end)
            self.prev_ppx[i] = self.ppx
            self.ppy[i] = np.random.uniform(self.span_y_start, self.span_y_end)
            self.prev_ppy[i] = self.ppy
            reint = True
            #drones to close, check
            while reint:
                reint = False
                for j in range(i):
                    distance = np.sqrt((self.ppx[j] - self.ppx[i])**2 + (self.ppy[j] - self.ppy[i])**2)
                    if distance < self.critical_radius:
                        reint = True
                if reint:
                    self.ppx[i] = np.random.uniform(self.span_x_start, self.span_x_end)
                    self.prev_ppx[i] = self.ppx
                    self.ppy[i] = np.random.uniform(self.span_y_start, self.span_y_end)
                    self.prev_ppy[i] = self.ppy
            self.pvx[i] = np.random.uniform(-self.max_velocity, self.max_velocity) * 0.8
            self.pvy[i] = np.random.uniform(-self.max_velocity, self.max_velocity) * 0.8
            self.tx[i] = np.random.uniform(self.span_x_start, self.span_x_end)
            self.ty[i] = np.random.uniform(self.span_y_start, self.span_y_end)
            self.tx_train[i] = np.random.uniform(self.span_x_start, self.span_x_end)
            self.ty_train[i] = np.random.uniform(self.span_y_start, self.span_y_end)
            self.orientation[i] = np.random.uniform(-np.pi, np.pi)
            self.state_to_history(i)
            # self.orientation[i] = 0.0

        self.t = 0
        for i in range(self.num_drones):
            self.update_states(i)
            states.append(self.get_state(i))

        return states

    def state_to_history(self, drone_id):
        self.ppx_history[drone_id].append(self.ppx[drone_id])
        self.ppy_history[drone_id].append(self.ppy[drone_id])
        self.pvx_history[drone_id].append(self.pvx[drone_id])
        self.pvy_history[drone_id].append(self.pvy[drone_id])
        self.tx_history[drone_id].append(self.tx[drone_id])
        self.ty_history[drone_id].append(self.ty[drone_id])
        self.orientation_history[drone_id].append(self.orientation[drone_id])

    def rotate_vel(self, x_vel, y_vel, angle):
        x = x_vel * np.cos(angle) - y_vel * np.sin(angle)
        y = x_vel * np.sin(angle) + y_vel * np.cos(angle)
        return x, y

    def particle_to_global(self, drone_id, particle):
        x, y = self.rotate_vel(particle.x, particle.y, self.orientation[drone_id])
        yaw = self.wrap_angle(self.orientation[drone_id] + particle.yaw)
        x += self.ppx[drone_id]
        y += self.ppy[drone_id]
        return x, y, yaw

    def set_goal_partices(self, drone_id, particles):
        self.particles_goal[drone_id] = []
        for p in particles:
            x, y, yaw = self.particle_to_global(drone_id, p)
            self.particles_goal[drone_id].append([x, y, yaw])

    def set_obs_partices(self, drone_id, particles):
        self.particles_obs[drone_id] = []
        for p in particles:
            x, y, yaw = self.particle_to_global(drone_id, p)
            self.particles_obs[drone_id].append([x, y, yaw])



    def load_cfg(self, nn_cfg, simulation_cfg):
        assert isinstance(simulation_cfg, SimulationConfig)
        # assert isinstance(nn_cfg, NNConfigMovement)

        self.max_velocity = nn_cfg.max_velocity
        self.speed_limit = simulation_cfg.speed_limit
        self.acceleration = nn_cfg.acceleration
        self.update_rate = nn_cfg.update_rate
        self.damping_per_sec = nn_cfg.damping_per_sec
        self.brake_per_sec = nn_cfg.brake_per_sec
        self.max_sensor_distance = nn_cfg.max_sensor_distance
        self.critical_radius = simulation_cfg.critical_radius
        self.goal_update_distance = simulation_cfg.goal_update_distance
        self.total_meter_x = simulation_cfg.total_meter_x
        self.wall_size_x = simulation_cfg.wall_size_x
        self.total_meter_y = simulation_cfg.total_meter_y
        self.wall_size_y = simulation_cfg.wall_size_y
        self.drone_radius = simulation_cfg.drone_radius
        self.goal_update_steps = simulation_cfg.goal_update_steps
        self.max_epLength = simulation_cfg.max_epLength

        self.span_x_start = - simulation_cfg.span_x / 2.0
        self.span_y_start = - simulation_cfg.span_y / 2.0
        self.span_x_end = simulation_cfg.span_x / 2.0
        self.span_y_end = simulation_cfg.span_y / 2.0
        self.slow_down_max = simulation_cfg.slow_down_max
        self.slow_down_min = simulation_cfg.slow_down_min
        self.slow_down_radius_max = simulation_cfg.slow_down_radius_max
        self.slow_down_radius_min = simulation_cfg.slow_down_radius_min
        self.num_drones = simulation_cfg.num_drones
        # actions = cfg['controlls']['actions']
        self.area_pixel_x = simulation_cfg.area_pixel_x
        self.debug_pixel_x = simulation_cfg.debug_pixel_x
        self.area_pixel_y = simulation_cfg.area_pixel_y
        self.debug_pixel_y = simulation_cfg.debug_pixel_y
        # self.x_actions_len = len(actions)
        # self.actions = self.generate_action(actions)
        # self.num_actions = len(self.actions) + 1
        self.ppx = [0.0] * self.num_drones
        self.prev_ppx = [0.0] * self.num_drones
        self.ppy = [0.0] * self.num_drones
        self.prev_ppy = [0.0] * self.num_drones
        self.pvx = [0.0] * self.num_drones
        self.pvy = [0.0] * self.num_drones

        self.ppx_history = [collections.deque(maxlen=HISTORY_LENGTH) for i in range(self.num_drones)]
        self.ppy_history = [collections.deque(maxlen=HISTORY_LENGTH) for i in range(self.num_drones)]
        self.pvx_history = [collections.deque(maxlen=HISTORY_LENGTH) for i in range(self.num_drones)]
        self.pvy_history = [collections.deque(maxlen=HISTORY_LENGTH) for i in range(self.num_drones)]
        self.tx_history = [collections.deque(maxlen=HISTORY_LENGTH) for i in range(self.num_drones)]
        self.ty_history = [collections.deque(maxlen=HISTORY_LENGTH) for i in range(self.num_drones)]
        self.orientation_history = [collections.deque(maxlen=HISTORY_LENGTH) for i in range(self.num_drones)]

        self.tx = [0.0] * self.num_drones
        self.tx_train = [0.0] * self.num_drones
        self.ty = [0.0] * self.num_drones
        self.ty_train = [0.0] * self.num_drones
        # self.velocity = [0.0] * self.num_drones
        # self.velocity_angle = [0.0] * self.num_drones
        # self.obstacle_angle = [None] * self.num_drones
        # self.obstacle_distance = [None] * self.num_drones
        self.goal_id = [-1] * self.num_drones
        self.obstacle_id = [-1] * self.num_drones
        self.goal_collected = [False] * self.num_drones
        self.goal_tick = [0] * self.num_drones
        self.crashed = [False] * self.num_drones
        self.goal_time_reset = [False] * self.num_drones

        self.orientation = [0.0] * self.num_drones
        self.num_actions = 10
        # self.num_actions = 5

    def generate_action(self, actions):
        len_actions = len(actions) * 2 if 0 not in actions else len(actions) * 2 - 1
        result = [0.0] * len_actions
        for idx, a in enumerate(actions):
            result[idx] = a
            if a != 0.0:
                result[idx + len(actions)] = a
        return result

    def get_num_actions(self):
        return self.num_actions

    def get_num_states(self):
        return self.num_states

    def closes_drone(self, my_id):
        if self.num_drones == 1:
            return None
        my_x = self.ppx[my_id]
        my_y = self.ppy[my_id]
        id = my_id
        d = 10000000.0
        for i in range(self.num_drones):
            if i == my_id:
                continue
            tmp_x = self.ppx[i]
            tmp_y = self.ppy[i]
            tmp = np.math.sqrt((tmp_x - my_x) ** 2 + (tmp_y - my_y) ** 2)
            if tmp < d:
                d = tmp
                id = i
        return id

    def get_distance_to(self, my_id, c_id, mean = None, variance = None):
        my_x = self.ppx[my_id]
        my_y = self.ppy[my_id]

        tmp_x = self.ppx[c_id]
        tmp_y = self.ppy[c_id]
        d = np.math.sqrt((tmp_x - my_x) ** 2 + (tmp_y - my_y) ** 2)
        if mean is not None and variance is not None:
            d += np.random.normal(mean, variance)
        return d

    def get_goal_distance(self, drone_id, mean=None, variance=None):
        dx = self.tx[drone_id] - self.ppx[drone_id]
        dy = self.ty[drone_id] - self.ppy[drone_id]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if mean is not None and variance is not None:
            distance += np.random.normal(mean, variance)
            distance = max(0.0, distance)
        return min(distance, self.max_sensor_distance)  # target

    def get_goal_angle(self, drone_id):
        dx = self.tx[drone_id] - self.ppx[drone_id]
        dy = self.ty[drone_id] - self.ppy[drone_id]
        return self.wrap_angle(np.math.atan2(dy, dx) - self.orientation[drone_id])

    def get_goal_distance_train(self, drone_id):
        dx = self.tx_train[drone_id] - self.ppx[drone_id]
        dy = self.ty_train[drone_id] - self.ppy[drone_id]
        return min(np.sqrt(dx ** 2 + dy ** 2), self.max_sensor_distance)  # target

    def get_goal_angle_train(self, drone_id):
        dx = self.tx_train[drone_id] - self.ppx[drone_id]
        dy = self.ty_train[drone_id] - self.ppy[drone_id]
        return self.wrap_angle(np.math.atan2(dy, dx) - self.orientation[drone_id])

    def get_velocity(self, drone_id):
        return np.sqrt(self.pvx[drone_id] ** 2 + self.pvy[drone_id] ** 2)

    def get_velocity_angle(self, drone_id):
        return self.wrap_angle(np.math.atan2(self.pvy[drone_id], self.pvx[drone_id]))


    def get_state(self, drone_id):

        update_tick = self.translate(1.0 / self.update_rate, 0.0, 1.0, self.nn_config.min_state_out,
                                     self.nn_config.max_state_out)

        gx = self.get_goal_distance(drone_id) * np.cos(self.get_goal_angle(drone_id))
        gy = self.get_goal_distance(drone_id) * np.sin(self.get_goal_angle(drone_id))

        ox = self.get_obstacle_distance(drone_id) * np.cos(self.get_obstacle_angle(drone_id))
        oy = self.get_obstacle_distance(drone_id) * np.sin(self.get_obstacle_angle(drone_id))

        # state = [velocity, velocity_angle, goal_distance, goal_angle]
        # print "x: {0} y: {1} goal_distance {2} goal_angle {3}".format(x, y, self.goal_distance[drone_id], self.goal_angle[drone_id])
        g_dx = self.translate(gx, -self.max_sensor_distance, self.max_sensor_distance,
                              self.nn_config.min_state_out, self.nn_config.max_state_out)
        g_dy = self.translate(gy, -self.max_sensor_distance, self.max_sensor_distance, self.nn_config.min_state_out,
                              self.nn_config.max_state_out)
        o_dx = self.translate(ox, -self.max_sensor_distance, self.max_sensor_distance,
                              self.nn_config.min_state_out, self.nn_config.max_state_out)
        o_dy = self.translate(oy, -self.max_sensor_distance, self.max_sensor_distance, self.nn_config.min_state_out,
                              self.nn_config.max_state_out)
        pvx = self.translate(self.pvx[drone_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)
        pvy = self.translate(self.pvy[drone_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)
        # o_pvx = 0.0
        # o_pvy = 0.0
        # c_id = self.closes_drone(drone_id)
        # if c_id is not None:
        #     o_pvx = self.translate(self.pvx[c_id], -self.max_velocity, self.max_velocity,
        #                          self.nn_config.min_state_out, self.nn_config.max_state_out)
        #     o_pvy = self.translate(self.pvy[c_id], -self.max_velocity, self.max_velocity,
        #                          self.nn_config.min_state_out, self.nn_config.max_state_out)

        #original
        return [pvx, pvy, g_dx, g_dy, o_dx, o_dy, update_tick]
        # return [pvx, pvy, o_pvx, o_pvy, g_dx, g_dy, o_dx, o_dy, update_tick]



    def get_goal_angle_translate(self, source_id, min_out, max_out):
        return self.translate(self.get_goal_angle(source_id), -np.pi, np.pi, min_out, max_out)

    def get_goal_angle_train_translate(self, source_id, min_out, max_out):
        return self.translate(self.get_goal_angle_train(source_id), -np.pi, np.pi, min_out, max_out)

    def get_obstacle_angle_translate(self, source_id, min_out, max_out):
        return self.translate(self.get_obstacle_angle(source_id), -np.pi, np.pi, min_out, max_out)

    # def get_random_goal_state(self, source_id, random_goal, min_output, max_output,  distance_mean=None, distance_variance=None):
    #     distance_noise = 0.0
    #     if distance_mean is not None and distance_noise is not None:
    #         distance_noise = np.random.normal(distance_mean, distance_variance)
    #         # print distance_noise
    #     angle, distance = self.drone_info_to_point(source_id, random_goal[0], random_goal[1])
    #     goal_distance = self.translate(distance + distance_noise, 0, self.max_sensor_distance, min_output,
    #                                    max_output)
    #     goal_angle = self.translate(angle, -np.pi, np.pi, min_output, max_output)
    #     velocity = self.translate(self.get_velocity(source_id), 0, self.max_velocity, min_output, max_output)
    #     velocity_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)
    #     # debug
    #     # print self.velocity[source_id], self.velocity_angle[source_id], 0, 0, self.goal_distance[
    #     #     source_id], 1.0 / self.update_rate
    #     # print "my_vel: {} my_angle: {} distance: {} dt: {}".format(self.velocity[source_id],
    #     #                                                            self.velocity_angle[source_id],
    #     #                                                            self.goal_distance[source_id],
    #     #                                                            1.0 / self.update_rate)
    #     # return [dx / d1, dy / d1]
    #     return [velocity, velocity_angle, 0.0, 0.0, goal_distance, 1.0 / self.update_rate]

    def get_goal_angle_state(self, source_id, min_output, max_output,  distance_mean=None, distance_variance=None):
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
            # print distance_noise
        #
        goal_distance = self.translate(self.get_goal_distance(source_id) + distance_noise, 0, self.max_sensor_distance, min_output,
                                       max_output)
        # velocity = self.translate(self.get_velocity(source_id), 0, self.max_velocity, min_output, max_output)
        # velocity_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)
        pvx = self.translate(self.pvx[source_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)
        pvy = self.translate(self.pvy[source_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)

        return [pvx, pvy, 0.0, 0.0, goal_distance, 1.0 / self.update_rate]

    def get_goal_angle_state_angle_vel(self, source_id, min_output, max_output,  distance_mean=None, distance_variance=None):
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
            # print distance_noise
        #
        goal_distance = self.translate(self.get_goal_distance(source_id) + distance_noise, 0, self.max_sensor_distance, min_output,
                                       max_output)
        velocity = self.translate(self.get_velocity(source_id), 0, self.max_velocity, min_output, max_output)
        velocity_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)

        return [velocity, velocity_angle, 0.0, 0.0, goal_distance, 1.0 / self.update_rate]

    def get_train_goal_angle_state(self, source_id, min_output, max_output,  distance_mean=None, distance_variance=None):
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
            # print distance_noise
        #
        goal_distance = self.translate(self.get_goal_distance_train(source_id) + distance_noise, 0, self.max_sensor_distance, min_output,
                                       max_output)
        # velocity = self.translate(self.get_velocity(source_id), 0, self.max_velocity, min_output, max_output)
        # velocity_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)
        pvx = self.translate(self.pvx[source_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)
        pvy = self.translate(self.pvy[source_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)

        return [pvx, pvy, 0.0, 0.0, goal_distance, 1.0 / self.update_rate]
        # return [pvx, pvy, goal_distance, 1.0 / self.update_rate]

    def get_train_goal_angle_state_angle_vel(self, source_id, min_output, max_output,  distance_mean=None, distance_variance=None):
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
            # print distance_noise
        #
        goal_distance = self.translate(self.get_goal_distance_train(source_id) + distance_noise, 0, self.max_sensor_distance, min_output,
                                       max_output)
        velocity = self.translate(self.get_velocity(source_id), 0, self.max_velocity, min_output, max_output)
        velocity_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)

        return [velocity, velocity_angle, 0.0, 0.0, goal_distance, 1.0 / self.update_rate]

    def get_obstacle_angle_state(self, source_id, min_output, max_output, distance_mean=None, distance_variance=None):
        c_id = self.closes_drone(source_id)
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
        # velocity = self.translate(self.get_velocity(source_id), 0, self.max_velocity, min_output, max_output)
        # velocity_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)
        pvx = self.translate(self.pvx[source_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)
        pvy = self.translate(self.pvy[source_id], -self.max_velocity, self.max_velocity,
                             self.nn_config.min_state_out, self.nn_config.max_state_out)
        obstacle_distance = self.translate(self.get_obstacle_distance(source_id) + distance_noise, 0, self.max_sensor_distance,
                                           min_output, max_output)
        # obstacle_angle = self.translate(self.obstacle_angle[source_id], -np.pi, np.pi, min_output, max_output)
        if c_id is None:
            ovx = self.translate(0.0, -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
            ovy = self.translate(0.0, -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
        else:
            ovx = self.translate(self.pvx[c_id], -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
            ovy = self.translate(self.pvy[c_id], -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
        # return [dx / d1, dy / d1]
        return [pvx, pvy, ovx, ovy, obstacle_distance, 1.0 / self.update_rate]
        # return [pvx, pvy, obstacle_distance, 1.0 / self.update_rate]

    def get_obstacle_angle_state_angle_vel(self, source_id, min_output, max_output, distance_mean=None, distance_variance=None):
        c_id = self.closes_drone(source_id)
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
        velocity = self.translate(self.get_velocity(source_id), 0, self.max_velocity, min_output, max_output)
        velocity_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)

        obstacle_distance = self.translate(self.get_obstacle_distance(source_id) + distance_noise, 0, self.max_sensor_distance,
                                           min_output, max_output)
        # obstacle_angle = self.translate(self.obstacle_angle[source_id], -np.pi, np.pi, min_output, max_output)
        if c_id is None:
            o_vel = self.translate(0.0, -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
            o_angle = self.translate(0.0, -np.pi, np.pi, min_output, max_output)
        else:
            o_vel = self.translate(self.get_velocity(c_id), -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
            o_angle = self.translate(self.get_velocity_angle(source_id), -np.pi, np.pi, min_output, max_output)

        # return [dx / d1, dy / d1]
        return [velocity, velocity_angle, o_vel, o_angle, obstacle_distance, 1.0 / self.update_rate]
        # return [velocity, velocity_angle, obstacle_angle]

    def translate(self, value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        output = rightMin + (valueScaled * rightSpan)
        output = max(output, rightMin)
        output = min(output, rightMax)

        return output


    def wrap_angle(self, angle):
        a = angle
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a

    def rotate_vel(self, x_vel, y_vel, angle):
        x = x_vel * np.cos(angle) - y_vel * np.sin(angle)
        y = x_vel * np.sin(angle) + y_vel * np.cos(angle)
        return x, y

    def cut_velocity(self, vx, vy, max_velo):
        norm = np.sqrt(vx ** 2 + vy ** 2)
        vx_c = vx
        vy_c = vy
        if norm > max_velo:
            vx_c *= max_velo / norm
            vy_c *= max_velo / norm
        return vx_c, vy_c

    def propotional_slow_down(self, drone_id):
        c_id = self.closes_drone(drone_id)
        if c_id < 0:
            return self.speed_limit
        distance = self.get_distance_to(drone_id, c_id)
        if distance < self.slow_down_radius_max:
            if distance < self.slow_down_radius_min:
                return self.slow_down_min
            tmp_d = distance - self.slow_down_radius_min
            r = self.slow_down_radius_max - self.slow_down_radius_min
            factor = tmp_d / r
            r = self.slow_down_max - self.slow_down_min
            return self.slow_down_min + r * factor
        else:
            return self.speed_limit

    def single_step(self, drone_id, action):
        # if self.goal_collected[drone_id] or self.t % (int(self.update_goal_step * self.update_rate)) == 0:
        if self.goal_collected[drone_id] or self.goal_tick[drone_id] % (int(self.update_goal_step * self.update_rate)) == 0:
        # if self.t % (self.update_goal_step * int(self.update_rate)) == 0:
            self.goal_tick[drone_id] = 0
            self.tx[drone_id] = np.random.uniform(self.span_x_start, self.span_x_end)
            self.ty[drone_id] = np.random.uniform(self.span_y_start, self.span_y_end)
            self.goal_id[drone_id] += 1
            self.goal_collected[drone_id] = False
            if self.goal_tick[drone_id] % (int(self.update_goal_step * self.update_rate)) == 0:
                self.goal_time_reset[drone_id] = True

        if self.crashed[drone_id]:
            print "reset Simulation!!!!"

        self.prev_ppx[drone_id] = self.ppx[drone_id]
        self.prev_ppy[drone_id] = self.ppy[drone_id]
        # convert velo to global_frame
        global_x_vel, global_y_vel = self.rotate_vel(self.pvx[drone_id] / self.update_rate,
                                                     self.pvy[drone_id] / self.update_rate,
                                                     self.orientation[drone_id])
        self.ppx[drone_id] += global_x_vel
        self.ppy[drone_id] += global_y_vel

        accel = self.acceleration_per_tick
        diagonal = np.sqrt((self.acceleration_per_tick ** 2) / 2.0)
        if action == 0:
            self.pvy[drone_id] -= accel
        elif action == 1:
            self.pvy[drone_id] += accel
        elif action == 2:
            self.pvx[drone_id] += accel
        elif action == 3:
            self.pvx[drone_id] -= accel
        elif action == 4:
            self.pvx[drone_id] += diagonal
            self.pvy[drone_id] += diagonal
        elif action == 5:
            self.pvx[drone_id] += diagonal
            self.pvy[drone_id] -= diagonal
        elif action == 6:
            self.pvx[drone_id] -= diagonal
            self.pvy[drone_id] += diagonal
        elif action == 7:
            self.pvx[drone_id] -= diagonal
            self.pvy[drone_id] -= diagonal
        elif action == 8:
            self.pvx[drone_id] *= self.brake_per_tick
            self.pvy[drone_id] *= self.brake_per_tick
        else:
            self.pvx[drone_id] *= self.damping_per_tick
            self.pvy[drone_id] *= self.damping_per_tick

        vel = self.propotional_slow_down(drone_id)
        self.pvx[drone_id], self.pvy[drone_id] = self.cut_velocity(self.pvx[drone_id], self.pvy[drone_id],
                                                                   vel)

        if self.ppx[drone_id] < self.x_start + self.drone_radius:
            self.pvx[drone_id] *= -1.0
            self.ppx[drone_id] = self.x_start + self.drone_radius + self.push_distance

        if self.ppx[drone_id] > self.x_end - self.drone_radius:
            self.pvx[drone_id] *= -1.0
            self.ppx[drone_id] = self.x_end - self.drone_radius - self.push_distance

        if self.ppy[drone_id] < self.y_start + self.drone_radius:
            self.pvy[drone_id] *= -1.0
            self.ppy[drone_id] = self.y_start + self.drone_radius + self.push_distance

        if self.ppy[drone_id] > self.y_end - self.drone_radius:
            self.pvy[drone_id] *= -1.0
            self.ppy[drone_id] = self.y_end - self.drone_radius - self.push_distance

        self.state_to_history(drone_id)

        return self.update_states(drone_id)


    def drone_info_to_point(self,drone_id, x, y):
        px = self.ppx[drone_id]
        py = self.ppy[drone_id]
        target_dx = x - px
        target_dy = y - py
        distance = min(np.sqrt(target_dx ** 2 + target_dy ** 2), self.max_sensor_distance)  # target
        angle = self.wrap_angle(np.math.atan2(target_dy, target_dx) - self.orientation[drone_id])
        return angle, distance

    def get_random_point(self):
        x = np.random.uniform(self.span_x_start, self.span_x_end)
        y = np.random.uniform(self.span_y_start, self.span_y_end)
        return [x, y]

    def get_obstacle_distance(self, drone_id):
        c_id = self.closes_drone(drone_id)
        if c_id is not None:
            tx2 = self.ppx[c_id]
            ty2 = self.ppy[c_id]
            # return [(tx - px), ty - py]
            obstacle_dx = tx2 - self.ppx[drone_id]
            obstacle_dy = ty2 - self.ppy[drone_id]
            return min(np.sqrt(obstacle_dx ** 2 + obstacle_dy ** 2), self.max_sensor_distance)
        else:
            return self.max_sensor_distance

    def get_obstacle_angle(self, drone_id):
        c_id = self.closes_drone(drone_id)
        if c_id is not None:
            tx2 = self.ppx[c_id]
            ty2 = self.ppy[c_id]
            # return [(tx - px), ty - py]
            obstacle_dx = tx2 - self.ppx[drone_id]
            obstacle_dy = ty2 - self.ppy[drone_id]
            return self.wrap_angle(np.math.atan2(obstacle_dy, obstacle_dx) - self.orientation[drone_id])
        else:
            return 0.0

    def update_states(self, drone_id):
        goal_collect = False
        crash = False
        if self.get_goal_distance(drone_id) < self.goal_update_distance:
            goal_collect = True
            if self.get_goal_distance(drone_id) < self.goal_update_distance:
                self.goal_count += 1

        if self.get_obstacle_distance(drone_id) <= self.drone_radius * 2:
            self.crash_count += 1
            crash = True

        return goal_collect, crash

    def calc_reward(self, actions, id):
        #goal distance reward
        c_id = self.closes_drone(id)
        if c_id is None or self.get_obstacle_distance(id) > self.critical_radius:
            r = -(self.get_goal_distance(id) / self.max_sensor_distance)
            if self.get_goal_distance(id) < self.goal_update_distance:
                r = -((self.get_goal_distance(id) / self.max_sensor_distance)**2)
            # if self.goal_distance[id] <= self.goal_update_distance and actions[id] == 8:
            #     r = 0.0
            # r -= self.translate(abs(self.wrap_angle(self.goal_angle[id] - self.velocity_angle[id])), 0, np.pi,
            #                      0.0, 1.0) * 0.25
        else:
            # r = (-self.goal_distance[id] / self.max_sensor_distance)
            r = -1.0




        if c_id is not None:
            d2 = self.get_obstacle_distance(id)
            # velocity reward
            # if d2 < self.slow_down_radius:
            #     if self.critical_radius == self.slow_down_radius:
            #         distance_factor = 0.0
            #     else:
            #         distance_factor = self.translate(d2, self.critical_radius, self.slow_down_radius, 0.0, 1.0)
            #     max_vel = distance_factor * (self.slow_down_max - self.slow_down_min) + self.slow_down_min
            #     my_vel = np.sqrt(self.pvx[id]**2 + self.pvy[id]**2)
            #     if my_vel > max_vel:
            #         r -= 1.0
            # obstacle_distance reward
            if d2 < self.critical_radius:
                # r += 2 * (d2 - self.critical_radius) / self.critical_radius
                r += 3 * (d2 - self.D) / self.critical_radius
                if d2 <= self.drone_radius * 2.0:
                    r -= 5.0
                    print "crash_rewards"

        return r



    def step(self, actions):
        if len(actions) != self.num_drones:
            pass
        self.t += 1
        rewards = [0] * self.num_drones
        new_states = [0] * self.num_drones

        for i in range(self.num_drones):
            self.goal_tick[i] += 1
            self.goal_collected[i], self.crashed[i] = self.single_step(i, actions[i])


        for i in range(self.num_drones):
            rewards[i] = self.calc_reward(actions, i)
            new_states[i] = self.get_state(i)

        # done = False

        return self.goal_collected, new_states, rewards, self.crashed

    def render(self, debug_data):
        if self.show:
            self.frontend.render(debug_data)
