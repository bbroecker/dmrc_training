import yaml

import numpy as np

from tf_nn_sim.Environments import Environment
from tf_nn_sim.Environments import PuckNormalRandomNoMoveFrontend

CONFIG_FILE = "../Config/PuckNormalRandomNoMove/puckworld.yaml"


class PuckNormalRandomworldNoMove(Environment):
    def __init__(self, show = True):

        Environment.__init__(self, CONFIG_FILE)
        self.max_velocity = None
        self.access = None
        self.damping = None
        self.damping_rate = None
        self.bad_radius = None
        self.wall_size_x = None
        self.wall_size_y = None
        self.actions = None
        self.x_actions_len = None
        self.num_actions = None
        self.drone_radius = None
        self.ppx = None
        self.prev_ppx = None
        self.ppy = None
        self.prev_ppy = None
        self.pvx = None
        self.pvy = None
        self.tx = None
        self.ty = None
        self.t = None
        self.update_rate = None
        self.angle_obstacle = []
        self.goal_update_time = None
        self.min_speed = None
        self.max_speed = None
        self.skip_frames = None
        self.pixel_x = None
        self.meter_x = None
        self.pixel_y = None
        self.meter_y = None
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.num_drones = None
        self.y_end = None
        self.max_goal_distance = None
        self.load_cfg(CONFIG_FILE)
        self.reset()
        self.num_states = len(self.get_state(0))
        self.show = show
        self.goal_count = 0
        if show:
            self.frontend = PuckNormalRandomNoMoveFrontend(self)


    def reset(self):
        self.x_start = 0.0
        self.x_end = 1.0
        self.y_start = 0.0
        self.y_end = 1.0
        self.goal_count = 0
        # self.ppx = np.random.random() * self.wall_size_x
        states = []
        for i in range(self.num_drones):
            self.ppx[i] = np.random.uniform(self.x_start, self.x_end)
            # print self.ppx[i]
            self.prev_ppx[i] = self.ppx
            self.ppy[i] = np.random.uniform(self.y_start, self.y_end)
            self.prev_ppy[i] = self.ppy
            self.pvx[i] = self.pvy[i] = 0
            if i == 0:
                while self.pvx[i] == 0.0 and self.pvy[i] == 0.0:
                    self.pvx[i] = np.random.uniform(self.min_speed, self.max_speed) * np.sign(np.random.uniform(-1.0, 1.0))
                    # self.pvx[i] = np.random.uniform(0.0, 1.0) * self.max_speed - (self.max_speed / 2.0)
                    # self.pvx[i] = np.random.uniform(0.0, 1.0) * 0.05 - 0.025
                    self.pvy[i] = np.random.uniform(self.min_speed, self.max_speed) * np.sign(np.random.uniform(-1.0, 1.0))
            # self.pvy[i] = np.random.uniform(0.0, 1.0) * self.max_speed - (self.max_speed / 2.0)
            # self.pvy[i] = np.random.uniform(0.0, 1.0) * self.max_speed - (self.max_speed / 2.0)
            # self.pvy[i] = np.random.uniform(0.0, 1.0) * 0.05 - 0.025

        self.t = 0
        for i in range(self.num_drones):
            states.append(self.get_state(i))

        return states

    def load_cfg(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                cfg = yaml.load(stream)['Environment']
            except yaml.YAMLError as exc:
                print(exc)

        self.max_velocity = cfg['dynamics']['max_velocity']
        self.access = cfg['dynamics']['access']
        self.skip_frames = cfg['world']['skip_frames']
        self.min_speed = cfg['world']['min_speed']
        self.max_speed = cfg['world']['max_speed']
        self.num_drones = cfg['world']['num_drones']
        self.update_rate = cfg['controlls']['update_rate']
        self.pixel_x = cfg['frontend']['pixel_x']
        self.pixel_y = cfg['frontend']['pixel_y']
        self.drone_radius = cfg['world']['drone_radius']
        # self.num_actions = len(self.actions) + 1
        self.ppx = [0.0] * self.num_drones
        self.prev_ppx = [0.0] * self.num_drones
        self.ppy = [0.0] * self.num_drones
        self.prev_ppy = [0.0] * self.num_drones
        self.pvx = [0.0] * self.num_drones
        self.pvy = [0.0] * self.num_drones




    def get_num_actions(self):
        return self.num_actions

    def get_num_states(self):
        return self.num_states

    def closes_drone(self, my_id):
        my_x = self.ppx[my_id]
        my_y = self.ppy[my_id]
        id = my_id
        d = 10000000.0
        for i in range(self.num_drones):
            if i == my_id:
                continue
            tmp_x = self.ppx[i]
            tmp_y = self.ppy[i]
            tmp = np.math.sqrt((tmp_x - my_x)**2 + (tmp_y - my_y)**2)
            if tmp < d:
                d = tmp
                id = i
        return id

    def get_angle(self, source_id, other_id):
        tx = self.ppx[other_id] - self.ppx[source_id]
        ty = self.ppy[other_id] - self.ppy[source_id]
        angle = np.math.atan2(ty, tx)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        # return self.translate(angle/np.pi, -1.0, 1.0, 0.0, 1.0)
        return angle/np.pi

    def translate(self, value, leftMin, leftMax, rightMin, rightMax):
        # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

        # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

        # Convert the 0-1 range into a value in the right range.
        return rightMin + (valueScaled * rightSpan)

    def get_angle_state(self, source_id, other_id):
        px = self.ppx[source_id]
        py = self.ppy[source_id]
        # pvx = self.pvx[source_id], -self.max_speed, self.max_speed, 0.0, 1.0)
        pvx = self.pvx[source_id]
        pvx_other = self.pvx[other_id]
        # pvy = self.translate(self.pvy[source_id], -self.max_speed, self.max_speed, 0.0, 1.0)
        pvy = self.pvy[source_id]
        pvy_other = self.pvy[other_id]
        tx = self.ppx[other_id]
        ty = self.ppy[other_id]
        # return [(tx - px), ty - py]
        dx = tx - px
        dy = ty - py

        d1 = np.sqrt(dx * dx+dy * dy) #target
        # d1 = self.translate(d1, 0.0, 1.41421, 0.0, 1.0)

        # return [dx / d1, dy / d1]
        return [pvx, pvy, d1]

    def get_angle_obstacle(self, drone_id):
        c_id = self.closes_drone(drone_id)
        tx = self.ppx[c_id] - self.ppx[drone_id]
        ty = self.ppy[c_id] - self.ppy[drone_id]
        angle = np.math.atan2(ty, tx)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle/np.pi

    def get_state(self, drone_id):
        c_id = self.closes_drone(drone_id)
        px = self.ppx[drone_id]
        py = self.ppy[drone_id]
        pvx = self.pvx[drone_id]
        pvy = self.pvy[drone_id]
        tx = self.ppx[c_id]
        ty = self.ppy[c_id]
        # return [(tx - px), ty - py]
        dx = self.ppx[drone_id] - tx
        dy = self.ppy[drone_id] - ty

        d1 = np.sqrt(dx * dx+dy * dy) #target


        # return [pvx, pvy, d1/self.max_distance, d2/self.max_distance]
        # return [px, py, pvx, pvy, tx - px, ty - py, tx2 - px, ty2 - py]
        return [px -0.5, py -0.5, pvx*10.0, pvy*10.0, tx - px, ty - py]
        # return [pvx * 10.0, pvy * 10.0, d1, d2]
        # return [pvx * 10.0, pvy * 10.0, d1]
        # return [self.ppx, self.ppy, self.pvx, self.pvy, self.tx - self.ppx, self.ty - self.ppy,
        #         self.tx2 - self.ppx, self.ty2 - self.ppy]
    #


        # return [pvx, pvy, d1]
        # return [px, py, d1]


    def single_step(self, id):
        self.prev_ppx[id] = self.ppx[id]
        self.prev_ppy[id] = self.ppy[id]
        self.ppx[id] += self.pvx[id] * (self.skip_frames + 1)
        self.ppy[id] += self.pvy[id] * (self.skip_frames + 1)

        if self.ppx[id] < self.drone_radius:
            self.pvx[id] *= -1.0
            self.ppx[id] = self.drone_radius

        if self.ppx[id] > 1.0 - self.drone_radius:
            self.pvx[id] *= -1.0
            self.ppx[id] = 1.0 - self.drone_radius

        if self.ppy[id] < self.drone_radius:
            self.pvy[id] *= -1.0
            self.ppy[id] = self.drone_radius

        if self.ppy[id] > 1.0 - self.drone_radius:
            self.pvy[id] *= -1.0
            self.ppy[id] = 1.0 - self.drone_radius

        # dx = self.ppx[id] - self.tx[id]
        # dy = self.ppy[id] - self.ty[id]
        # d1 = np.sqrt(dx * dx+dy * dy)

    def check_crash(self, id):
        done = False

        c_id = self.closes_drone(id)
        dx = self.ppx[id] - self.ppx[c_id]
        dy = self.ppy[id] - self.ppy[c_id]
        d = np.sqrt(dx * dx+dy * dy)

        if d < 0.1:
            done = True

        return done


    def step(self):
        self.t += 1

        for i in range(self.num_drones):
            self.single_step(i)

        new_states = [0] * self.num_drones
        done = [False] * self.num_drones

        for i in range(self.num_drones):
            done[i] = self.check_crash(i)
            new_states[i] = self.get_state(i)

        # done = False

        return new_states, done

    def render(self):
        if self.show:
            self.frontend.render()