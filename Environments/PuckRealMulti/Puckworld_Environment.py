import numpy as np

from tf_nn_sim.Config.NNConfigMovement import NNConfigMovement
from tf_nn_sim.Config.SimConfig import SimulationConfig
from tf_nn_sim.Environments.Environment import Environment
from tf_nn_sim.Environments.PuckRealMulti.Frontend_Puck import PuckRealMultiFrontend

CONFIG_FILE = "../Config/PuckRealMulti/puckworld.yaml"

# self.nn_config.min_state_out = -1.0
# self.nn_config.max_state_out = 1.0
# self.nn_config.min_reward = -1.0
# self.nn_config.max_reward = 1.0
# self.nn_config.min_state_out = -5.0
# self.nn_config.max_state_out = 5.0
# self.nn_config.min_reward = -1.0
# self.nn_config.max_reward = 1.0

TRUNC_TIME = 1.0
LEFT_PREF = 0.5
COLLISION_BUFFER = 0.20


class PuckRealMultiworld(Environment):
    def __init__(self, nn_config, simulation_cfg, show=True):
        assert isinstance(simulation_cfg, SimulationConfig)
        # assert isinstance(nn_config, NNConfigMovement)
        Environment.__init__(self, CONFIG_FILE)
        self.max_velocity = None
        self.acceleration = None
        self.damping_per_sec = None
        self.critical_radius = None
        self.wall_size_x = None
        self.wall_size_y = None
        self.actions = None
        self.x_actions_len = None
        self.num_actions = None
        self.drone_radius = None
        self.orientation = None
        self.ppx = None
        self.prev_ppx = None
        self.ppy = None
        self.prev_ppy = None
        self.pvx = None
        self.pvy = None
        self.tx = None
        self.ty = None
        self.t = None
        self.nn_config = nn_config
        self.update_rate = None
        self.predict_obstacle_angle = []
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
        self.num_drones = None
        self.y_end = None
        self.max_sensor_distance = None
        self.push_distance = 0.1

        self.load_cfg(nn_config, simulation_cfg)
        # print self.max_sensor_distance
        # damping factor
        self.damping_per_tick = 1.0 - self.damping_per_sec / self.update_rate
        self.damping_per_tick = 0.0 if self.damping_per_tick < 0.0 else self.damping_per_tick
        self.num_states = len(self.get_state(0))
        self.acceleration_per_tick = self.acceleration * 1.0 / self.update_rate
        self.update_goal_step = self.goal_update_steps
        self.show = show
        self.goal_count = 0
        self.crash_count = 0
        self.reset()
        if show:
            self.frontend = PuckRealMultiFrontend(self)

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
        for i in range(self.num_drones):
            self.ppx[i] = np.random.uniform(self.x_start, self.x_end)
            self.prev_ppx[i] = self.ppx
            self.ppy[i] = np.random.uniform(self.y_start, self.y_end)
            self.prev_ppy[i] = self.ppy
            reint = True
            while reint:
                reint = False
                for j in range(i):
                    distance = np.sqrt((self.ppx[j] - self.ppx[i])**2 + (self.ppy[j] - self.ppy[i])**2)
                    if distance < self.critical_radius:
                        reint = True
                if reint:
                    self.ppx[i] = np.random.uniform(self.x_start, self.x_end)
                    self.prev_ppx[i] = self.ppx
                    self.ppy[i] = np.random.uniform(self.y_start, self.y_end)
                    self.prev_ppy[i] = self.ppy
            self.pvx[i] = np.random.uniform(-self.max_velocity, self.max_velocity) * 0.8
            self.pvy[i] = np.random.uniform(-self.max_velocity, self.max_velocity) * 0.8
            self.tx[i] = np.random.uniform(self.x_start + 0.2, self.x_end - 0.2)
            self.ty[i] = np.random.uniform(self.y_start + 0.2, self.y_end - 0.2)
            self.orientation[i] = np.random.uniform(-np.pi, np.pi)
            # self.orientation[i] = 0.0

        self.t = 0
        for i in range(self.num_drones):
            self.update_states(i)
            states.append(self.get_state(i))
            self.prev_goal_distance[i] = self.goal_distance[i]

        return states

    def load_cfg(self, nn_cfg, simulation_cfg):
        assert isinstance(simulation_cfg, SimulationConfig)
        # assert isinstance(nn_cfg, NNConfigMovement)

        self.max_velocity = nn_cfg.max_velocity
        self.acceleration = nn_cfg.acceleration
        self.update_rate = nn_cfg.update_rate
        self.damping_per_sec = nn_cfg.damping_per_sec
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
        self.tx = [0.0] * self.num_drones
        self.ty = [0.0] * self.num_drones
        self.goal_distance = [0.0] * self.num_drones
        self.prev_goal_distance = [0.0] * self.num_drones
        self.goal_angle = [0.0] * self.num_drones
        self.velocity = [0.0] * self.num_drones
        self.velocity_angle = [0.0] * self.num_drones
        self.obstacle_angle = [None] * self.num_drones
        self.obstacle_distance = [None] * self.num_drones

        self.orientation = [0.0] * self.num_drones
        self.num_actions = 9
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

    def get_state(self, drone_id):

        goal_distance = self.translate(self.goal_distance[drone_id], 0, self.max_sensor_distance,
                                       self.nn_config.min_state_out,
                                       self.nn_config.max_state_out)
        goal_angle = self.translate(self.goal_angle[drone_id], -np.pi, np.pi, self.nn_config.min_state_out,
                                    self.nn_config.max_state_out)
        velocity = self.translate(self.velocity[drone_id], 0, self.max_velocity, self.nn_config.min_state_out,
                                  self.nn_config.max_state_out)
        velocity_angle = self.translate(self.velocity_angle[drone_id], -np.pi, np.pi, self.nn_config.min_state_out,
                                        self.nn_config.max_state_out)
        update_tick = self.translate(1.0 / self.update_rate, 0.0, 1.0, self.nn_config.min_state_out,
                                     self.nn_config.max_state_out)
        # print "--------start------"
        # print "g_d {} g_a{} v {} v_a {}".format(self.goal_distance[drone_id], self.goal_angle[drone_id], self.velocity[drone_id],
        #                            self.velocity_angle[drone_id])
        # print "g_d {} g_a{} v {} v_a {}".format(goal_distance, goal_angle, velocity,
        #                            velocity_angle)
        # print "--------end------"


        if self.obstacle_distance[drone_id] is not None:
            c_id = self.closes_drone(drone_id)
            obstacle_distance = self.translate(self.obstacle_distance[drone_id], 0, self.max_sensor_distance,
                                               self.nn_config.min_state_out, self.nn_config.max_state_out)
            obstacle_angle = self.translate(self.obstacle_angle[drone_id], -np.pi, np.pi, self.nn_config.min_state_out,
                                            self.nn_config.max_state_out)
            gx = self.goal_distance[drone_id] * np.cos(self.goal_angle[drone_id])
            gy = self.goal_distance[drone_id] * np.sin(self.goal_angle[drone_id])
            ox = self.obstacle_distance[drone_id] * np.cos(self.obstacle_angle[drone_id])
            oy = self.obstacle_distance[drone_id] * np.sin(self.obstacle_angle[drone_id])
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
            o_distance = self.translate(self.obstacle_distance[drone_id], 0.0, self.max_sensor_distance, self.nn_config.min_state_out, self.nn_config.max_state_out)
            g_distance = self.translate(self.goal_distance[drone_id], 0.0, self.max_sensor_distance, self.nn_config.min_state_out, self.nn_config.max_state_out)

            return [pvx, pvy, g_distance, o_distance, update_tick]
            #original
            #return [pvx, pvy, g_dx, g_dy, o_dx, o_dy, update_tick]
            # return [velocity, velocity_angle, goal_distance, goal_angle, obstacle_distance, obstacle_angle]
            # return [velocity, velocity_angle, goal_distance, goal_angle, obstacle_distance, obstacle_angle]
            # return [pvx*10.0, pvy*10.0, tx - px, ty - py, tx2 - px, ty2 - py]
        else:
            # tmp = self.translate(self.wrap_angle(self.goal_angle[drone_id] - self.velocity_angle[drone_id]), -np.pi, np.pi, self.nn_config.min_state_out, self.nn_config.max_state_out)
            x = self.goal_distance[drone_id] * np.cos(self.goal_angle[drone_id])
            y = self.goal_distance[drone_id] * np.sin(self.goal_angle[drone_id])
            ox = self.max_sensor_distance * np.cos(0.0)
            oy = self.max_sensor_distance * np.sin(0.0)
            # state = [velocity, velocity_angle, goal_distance, goal_angle]
            # print "x: {0} y: {1} goal_distance {2} goal_angle {3}".format(x, y, self.goal_distance[drone_id], self.goal_angle[drone_id])
            dx = self.translate(x, -self.max_sensor_distance, self.max_sensor_distance,
                                self.nn_config.min_state_out, self.nn_config.max_state_out)
            dy = self.translate(y, -self.max_sensor_distance, self.max_sensor_distance, self.nn_config.min_state_out,
                                self.nn_config.max_state_out)
            o_dx = self.translate(ox, -self.max_sensor_distance, self.max_sensor_distance,
                                  self.nn_config.min_state_out, self.nn_config.max_state_out)
            o_dy = self.translate(oy, -self.max_sensor_distance, self.max_sensor_distance, self.nn_config.min_state_out,
                                  self.nn_config.max_state_out)
            pvx = self.translate(self.pvx[drone_id], -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
            pvy = self.translate(self.pvy[drone_id], -self.max_velocity, self.max_velocity,
                                 self.nn_config.min_state_out, self.nn_config.max_state_out)
            px = self.translate(self.ppx[drone_id], -self.wall_size_x / 2.0, self.wall_size_x / 2.0,
                                self.nn_config.min_state_out, self.nn_config.max_state_out)
            py = self.translate(self.ppy[drone_id], -self.wall_size_y / 2.0, self.wall_size_y / 2.0,
                                self.nn_config.min_state_out, self.nn_config.max_state_out)
            o_distance = self.translate(self.max_sensor_distance, 0.0, self.max_sensor_distance, self.nn_config.min_state_out, self.nn_config.max_state_out)
            g_distance = self.translate(self.goal_distance[drone_id], 0.0, self.max_sensor_distance, self.nn_config.min_state_out, self.nn_config.max_state_out)
            # state = [px, py, pvx, pvy, dx, dy]
            return [pvx, pvy, g_distance, o_distance, update_tick]
            #original
            # state = [pvx, pvy, dx, dy, o_dx, o_dy, update_tick]
            # return state

    def get_goal_angle(self, source_id, min_out, max_out):
        return self.translate(self.goal_angle[source_id], -np.pi, np.pi, min_out, max_out)

    def get_obstacle_angle(self, source_id, min_out, max_out):
        return self.translate(self.obstacle_angle[source_id], -np.pi, np.pi, min_out, max_out)

    def get_random_goal_state(self, source_id, random_goal, min_output, max_output,  distance_mean=None, distance_variance=None):
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
            # print distance_noise
        angle, distance = self.drone_info_to_point(source_id, random_goal[0], random_goal[1])
        goal_distance = self.translate(distance + distance_noise, 0, self.max_sensor_distance, min_output,
                                       max_output)
        goal_angle = self.translate(angle, -np.pi, np.pi, min_output, max_output)
        velocity = self.translate(self.velocity[source_id], 0, self.max_velocity, min_output, max_output)
        velocity_angle = self.translate(self.velocity_angle[source_id], -np.pi, np.pi, min_output, max_output)
        # debug
        # print self.velocity[source_id], self.velocity_angle[source_id], 0, 0, self.goal_distance[
        #     source_id], 1.0 / self.update_rate
        # print "my_vel: {} my_angle: {} distance: {} dt: {}".format(self.velocity[source_id],
        #                                                            self.velocity_angle[source_id],
        #                                                            self.goal_distance[source_id],
        #                                                            1.0 / self.update_rate)
        # return [dx / d1, dy / d1]
        return [velocity, velocity_angle, 0.0, 0.0, goal_distance, 1.0 / self.update_rate]

    def get_goal_angle_state(self, source_id, min_output, max_output,  distance_mean=None, distance_variance=None):
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
            # print distance_noise
        #
        goal_distance = self.translate(self.goal_distance[source_id] + distance_noise, 0, self.max_sensor_distance, min_output,
                                       max_output)
        goal_angle = self.translate(self.goal_angle[source_id], -np.pi, np.pi, min_output, max_output)
        velocity = self.translate(self.velocity[source_id], 0, self.max_velocity, min_output, max_output)
        velocity_angle = self.translate(self.velocity_angle[source_id], -np.pi, np.pi, min_output, max_output)
        # debug
        # print self.velocity[source_id], self.velocity_angle[source_id], 0, 0, self.goal_distance[
        #     source_id], 1.0 / self.update_rate
        # print "my_vel: {} my_angle: {} distance: {} dt: {}".format(self.velocity[source_id],
        #                                                            self.velocity_angle[source_id],
        #                                                            self.goal_distance[source_id],
        #                                                            1.0 / self.update_rate)
        # return [dx / d1, dy / d1]
        return [velocity, velocity_angle, 0.0, 0.0, goal_distance, 1.0 / self.update_rate]

    def get_obstacle_angle_state(self, source_id, min_output, max_output, distance_mean=None, distance_variance=None):
        c_id = self.closes_drone(source_id)
        distance_noise = 0.0
        if distance_mean is not None and distance_noise is not None:
            distance_noise = np.random.normal(distance_mean, distance_variance)
        velocity = self.translate(self.velocity[source_id], 0, self.max_velocity, min_output, max_output)
        velocity_angle = self.translate(self.velocity_angle[source_id], -np.pi, np.pi, min_output, max_output)
        obstacle_distance = self.translate(self.obstacle_distance[source_id] + distance_noise, 0, self.max_sensor_distance,
                                           min_output, max_output)
        # obstacle_angle = self.translate(self.obstacle_angle[source_id], -np.pi, np.pi, min_output, max_output)
        if c_id is None:
            obs_velocity = self.translate(0.0, 0.0, self.max_velocity, min_output, max_output)
            obs_velocity_angle = self.translate(0.0, -np.pi, np.pi, min_output, max_output)
        else:
            obs_velocity = self.translate(self.velocity[c_id], 0, self.max_velocity, min_output, max_output)
            obs_velocity_angle = self.translate(self.velocity_angle[c_id], -np.pi, np.pi, min_output, max_output)
        # return [dx / d1, dy / d1]
        return [velocity, velocity_angle, obs_velocity, obs_velocity_angle, obstacle_distance, 1.0 / self.update_rate]
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

    def get_obstacle_state(self, drone_id):
        px = self.ppx
        py = self.ppy
        pvx = self.pvx
        pvy = self.pvy
        tx2 = self.tx2
        ty2 = self.ty2

        dx2 = self.ppx - self.tx2
        dy2 = self.ppy - self.ty2

        d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)  # bad
        # return [self.tx2 - self.ppx, self.ty2 - self.ppy]
        return [pvx, pvy, d2, 0.95]
        # return [px, py, pvx, pvy, d2, 0.95]
        # return [px, py, d2]

    def wrap_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def rotate_vel(self, x_vel, y_vel, angle):
        x = x_vel * np.cos(angle) - y_vel * np.sin(angle)
        y = x_vel * np.sin(angle) + y_vel * np.cos(angle)
        return x, y

    def cut_velocity(self, vx, vy, max_velo):
        norm = np.sqrt(vx ** 2 + vy ** 2)
        if norm > max_velo:
            vx *= max_velo / norm
            vy *= max_velo / norm
        return vx, vy

    def single_step(self, drone_id, action):
        self.prev_ppx[drone_id] = self.ppx[drone_id]
        self.prev_ppy[drone_id] = self.ppy[drone_id]
        # convert velo to global_frame
        global_x_vel, global_y_vel = self.rotate_vel(self.pvx[drone_id] / self.update_rate,
                                                     self.pvy[drone_id] / self.update_rate,
                                                     self.orientation[drone_id])
        self.ppx[drone_id] += global_x_vel
        self.ppy[drone_id] += global_y_vel

        self.pvx[drone_id] *= self.damping_per_tick
        self.pvy[drone_id] *= self.damping_per_tick
        # accel = 0.004
        accel = self.acceleration_per_tick
        diagonal = np.sqrt((self.acceleration_per_tick ** 2) / 2.0)
        # accel = 0.004
        if action == 0:
            self.pvy[drone_id] -= accel
        if action == 1:
            self.pvy[drone_id] += accel
        if action == 2:
            self.pvx[drone_id] += accel
        if action == 3:
            self.pvx[drone_id] -= accel
        if action == 4:
            self.pvx[drone_id] += diagonal
            self.pvy[drone_id] += diagonal
        if action == 5:
            self.pvx[drone_id] += diagonal
            self.pvy[drone_id] -= diagonal
        if action == 6:
            self.pvx[drone_id] -= diagonal
            self.pvy[drone_id] += diagonal
        if action == 7:
            self.pvx[drone_id] -= diagonal
            self.pvy[drone_id] -= diagonal


        self.pvx[drone_id], self.pvy[drone_id] = self.cut_velocity(self.pvx[drone_id], self.pvy[drone_id],
                                                                   self.max_velocity)

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
        x = np.random.uniform(self.x_start + 0.2, self.x_end - 0.2)
        y = np.random.uniform(self.y_start + 0.2, self.y_end - 0.2)
        return [x, y]

    def update_states(self, drone_id):
        goal_update = False
        if self.t % (self.update_goal_step * int(self.update_rate)) == 0 or self.goal_distance[drone_id] < self.goal_update_distance:
        # if self.t % (self.update_goal_step * int(self.update_rate)) == 0:
            self.tx[drone_id] = np.random.uniform(self.x_start + 0.2, self.x_end - 0.2)
            self.ty[drone_id] = np.random.uniform(self.y_start + 0.2, self.y_end - 0.2)
            goal_update = True
            if self.goal_distance[drone_id] < self.goal_update_distance:
                self.goal_count += 1

        c_id = self.closes_drone(drone_id)
        self.prev_goal_distance[drone_id] = self.goal_distance[drone_id]
        px = self.ppx[drone_id]
        py = self.ppy[drone_id]
        pvx = self.pvx[drone_id]
        pvy = self.pvy[drone_id]
        tx = self.tx[drone_id]
        ty = self.ty[drone_id]
        target_dx = tx - px
        target_dy = ty - py
        self.goal_distance[drone_id] = min(np.sqrt(target_dx ** 2 + target_dy ** 2), self.max_sensor_distance)  # target
        self.goal_angle[drone_id] = self.wrap_angle(np.math.atan2(target_dy, target_dx) - self.orientation[drone_id])
        self.velocity[drone_id] = np.sqrt(pvx ** 2 + pvy ** 2)
        self.velocity_angle[drone_id] = self.wrap_angle(np.math.atan2(pvy, pvx))

        if c_id is not None:
            tx2 = self.ppx[c_id]
            ty2 = self.ppy[c_id]
            # return [(tx - px), ty - py]
            obstacle_dx = tx2 - self.ppx[drone_id]
            obstacle_dy = ty2 - self.ppy[drone_id]
            self.obstacle_angle[drone_id] = self.wrap_angle(
                np.math.atan2(obstacle_dy, obstacle_dx) - self.orientation[drone_id])
            self.obstacle_distance[drone_id] = min(np.sqrt(obstacle_dx ** 2 + obstacle_dy ** 2),
                                                   self.max_sensor_distance)
        else:
            self.obstacle_angle[drone_id] = 0.0
            self.obstacle_distance[drone_id] = self.max_sensor_distance

        return goal_update

    def calc_reward(self, actions, id):
        done = False
        # compute distances
        dx = self.ppx[id] - self.tx[id]
        dy = self.ppy[id] - self.ty[id]
        d1 = np.sqrt(dx * dx + dy * dy)
        d1 = min(d1, self.max_sensor_distance)
        # d1 = abs(dx) + abs(dy)

        # if d1 <self.bad_radius:
        #     r = -(d1**2)
        # else:
        # r = (-self.goal_distance[id] / self.max_sensor_distance)
        c_id = self.closes_drone(id)
        if c_id is None or self.obstacle_distance[id] > self.critical_radius:
            r = (-self.goal_distance[id] / self.max_sensor_distance)
            # if self.goal_distance[id] <= self.goal_update_distance and actions[id] == 8:
            #     r = 0.0
            # r -= self.translate(abs(self.wrap_angle(self.goal_angle[id] - self.velocity_angle[id])), 0, np.pi,
            #                      0.0, 1.0) * 0.25
        else:
            r = -1.0
        # print "reward {0} distance{1}".format(r, self.goal_distance)
        # r = -(self.goal_distance[id]) / self.max_sensor_distance
        # r -= self.translate(abs(self.wrap_angle(self.goal_angle[id] - self.velocity_angle[id])), 0, np.pi,
        #                      0.0, 1.0) * 0.25
        # closure_speed_goal = (self.goal_distance[id] - self.prev_goal_distance[id]) / (1.0 / self.update_rate)
        # r += self.translate(closure_speed_goal, -self.max_velocity, self.max_velocity, -1.0, 0.0) * 0.20
        # r = self.translate(r, -1.0, 0.0, -1.0, 1.0)

        # print " distance {0} reward {1}".format(-d1, r)
        # print "reward {0}".format(r)
        # if d1 >= 0.05 and actions[id] == self.num_actions - 1:
        #     print "close enough"
        # r = - 0.2


        d2 = 1000000.0
        # if c_id is not None:
        #     print "distance {0}".format(self.vo_distance(id, c_id))
        if c_id is not None:
            dx = self.ppx[id] - self.ppx[c_id]
            dy = self.ppy[id] - self.ppy[c_id]
            d2 = np.sqrt(dx * dx + dy * dy)
            # vio_dist = self.translate(self.vo_distance(id, c_id), -0.5, 0.0, 0.0, 1.0)
            # if math.isnan(vio_dist) or math.isinf(vio_dist):
            #     vio_dist = 1.0
            # r -= 1.0 - vio_dist
            if d2 < self.critical_radius:
                # tmp = (d2 - self.critical_radius) / self.critical_radius
                # tmp = max(-1.0, tmp * 5)
                # r += 3 * tmp
                # r += self.translate(abs(self.wrap_angle(self.obstacle_angle[id] - self.velocity_angle[id])), 0, np.pi,
                #                     -1.0, 0.0)
                r += 2 * (d2 - self.critical_radius) / self.critical_radius
                # if d2 < 0.1:
                #     r = -5.0
                #     done = True

        # if self.t % self.update_goal_step == 0 or self.goal_distance[id] <= self.goal_update_distance:
            # if d1 <= 0.1:
            #     self.goal_count += 1
        if self.obstacle_distance[id] <= self.drone_radius * 2:
            self.crash_count += 1
        # r = self.translate(r, -3.0, 0.0, self.nn_config.min_reward, self.nn_config.max_reward)
        # r = self.translate(r, -1.0, 0.0, self.nn_config.min_reward, self.nn_config.max_reward)
        # r = self.translate(r, -2.0, 0.0, self.nn_config.min_reward, self.nn_config.max_reward)
        if self.goal_distance[id] <= self.goal_update_distance or (
                        c_id is not None and self.obstacle_distance[id] <= self.drone_radius):
            if c_id is not None and self.obstacle_distance[id] <= self.drone_radius:
                pass
                # r = -5.0
            elif self.obstacle_distance[id] > self.critical_radius and self.goal_distance[
                id] <= self.goal_update_distance:
                # r = 0.0
                pass
            done = True

        return r, done

    # def vo_distance(self, drone_id1, drone_id2):
    #
    #     vel_1_x, vel_1_y = self.rotate_vel(self.pvx[drone_id1], self.pvx[drone_id1], self.orientation[drone_id1])
    #     vel_2_x, vel_2_y = self.rotate_vel(self.pvx[drone_id1], self.pvx[drone_id1], self.orientation[drone_id1])
    #
    #     vel_1 = np.array([vel_1_x, vel_1_y])
    #     pose_1 = np.array([self.ppx[drone_id1], self.ppy[drone_id1]])
    #
    #     vel_2 = np.array([vel_2_x, vel_2_y])
    #     pose_2 = np.array([self.ppx[drone_id2], self.ppy[drone_id2]])
    #
    #     combined_radius = self.drone_radius * 2.0
    #
    #     vo = create_vo(combined_radius + COLLISION_BUFFER, pose_1, vel_1, pose_2, vel_2,
    #                    trunc_time=TRUNC_TIME, left_pref=LEFT_PREF, vo_type=HRVO_TYPE)
    #
    #     distance = min_distance_to_vo(vo, vel_1)
    #     if not outside_cone(vo, vel_1):
    #         distance *= -1
    #
    #     return distance

    def step(self, actions):
        if len(actions) != self.num_drones:
            pass
        self.t += 1
        goal_update = False
        for i in range(self.num_drones):
            goal_update |= self.single_step(i, actions[i])

        rewards = [0] * self.num_drones
        new_states = [0] * self.num_drones
        done = [False] * self.num_drones

        for i in range(self.num_drones):
            rewards[i], done[i] = self.calc_reward(actions, i)
            new_states[i] = self.get_state(i)

        # done = False

        return goal_update, new_states, rewards, done

    def render(self, debug_data):
        if self.show:
            self.frontend.render(debug_data)
