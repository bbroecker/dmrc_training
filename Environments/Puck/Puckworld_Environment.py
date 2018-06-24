import yaml

import numpy as np
from Environments.Environment import Environment

from tf_nn_sim.Environments.Puck.Frontend_Puck import PuckFrontend

CONFIG_FILE = "Config/Puck/puckworld.yaml"


class Puckworld(Environment):
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
        self.angle_goal = None
        self.angle_obstacle = None
        self.prev_ppx = None
        self.prev_ppy = None
        self.ppx = None
        self.ppy = None
        self.pvx = None
        self.pvy = None
        self.tx = None
        self.ty = None
        self.tx2 = None
        self.ty2 = None
        self.t = None
        self.update_rate = None
        self.goal_update_time = None
        self.bad_speed = None
        self.pixel_x = None
        self.meter_x = None
        self.pixel_y = None
        self.meter_y = None
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.y_end = None
        self.max_goal_distance = None
        self.load_cfg(CONFIG_FILE)
        self.max_distance = np.sqrt(self.wall_size_x**2 + self.wall_size_y**2)
        self.damping_per_step = self.update_rate/self.damping_rate
        self.reset()
        self.num_states = len(self.get_state())
        self.access_per_step = self.access * 1.0 / self.update_rate
        self.update_goal_step = self.update_rate * self.goal_update_time
        self.bad_distance_step = self.bad_speed * 1.0/self.update_rate
        self.show = show
        if show:
            self.frontend = PuckFrontend(self)


    def reset(self):
        self.x_start = -self.wall_size_x / 2.0
        self.x_end = self.wall_size_x / 2.0
        self.y_start = -self.wall_size_y / 2.0
        self.y_end = self.wall_size_y / 2.0
        # self.ppx = np.random.random() * self.wall_size_x
        self.ppx = np.random.uniform(self.x_start, self.x_end)
        self.prev_ppx = self.ppx
        self.ppy = np.random.uniform(self.y_start, self.y_end)
        self.prev_ppy = self.ppy
        self.pvx = 0.0
        self.pvy = 0.0
        self.tx = np.random.uniform(self.x_start, self.x_end)
        self.ty = np.random.uniform(self.y_start, self.y_end)
        self.tx2 = np.random.uniform(self.x_start, self.x_end)
        self.ty2 = np.random.uniform(self.y_start, self.y_end)
        self.t = 0
        return self.get_state()

    def load_cfg(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                cfg = yaml.load(stream)['Environment']
            except yaml.YAMLError as exc:
                print(exc)

        self.max_velocity = cfg['dynamics']['max_velocity']
        self.access = cfg['dynamics']['access']
        self.damping = cfg['dynamics']['damping']
        self.damping_rate = cfg['dynamics']['damping_rate']
        self.bad_radius = cfg['world']['bad_radius']
        self.drone_radius = cfg['world']['drone_radius']
        self.wall_size_x = cfg['world']['wall_size_x']
        self.wall_size_y = cfg['world']['wall_size_y']
        self.goal_update_time = cfg['world']['goal_update_time']
        self.bad_speed = cfg['world']['bad_speed']
        self.max_goal_distance = cfg['world']['max_goal_distance']
        actions = cfg['controlls']['actions']
        self.update_rate = cfg['controlls']['update_rate']
        self.pixel_x = cfg['frontend']['pixel_x']
        self.meter_x = cfg['frontend']['meter_x']
        self.pixel_y = cfg['frontend']['pixel_y']
        self.meter_y = cfg['frontend']['meter_y']
        self.x_actions_len = len(actions)
        self.actions = self.generate_action(actions)
        self.num_actions = len(self.actions) + 1
        # self.num_actions = 3

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

    def get_state(self):
        px = self.ppx / (self.wall_size_x / 2.0)
        py = self.ppy / (self.wall_size_y / 2.0)
        pvx = self.pvx / self.max_velocity
        pvy = self.pvy / self.max_velocity
        tx = self.tx / (self.wall_size_x / 2.0)
        ty = self.ty / (self.wall_size_y / 2.0)
        tx2 = self.tx2 / (self.wall_size_x / 2.0)
        ty2 = self.ty2 / (self.wall_size_y / 2.0)
        # return [(tx - px), ty - py]
        dx = self.ppx - self.tx
        dy = self.ppy - self.ty
        d1 = np.sqrt(dx * dx+dy * dy) #target

        dx = (self.ppx - self.tx2)/self.max_distance
        dy = (self.ppy - self.ty2)/self.max_distance
        d2 = np.sqrt(dx * dx+dy * dy) #bad

        # return [pvx, pvy, d1/self.max_distance, d2/self.max_distance]
        return [pvx, pvy, d1/self.max_distance]
        # return [px, py, pvx, pvy, tx - px, ty - py,
        #         tx2 - px, ty2 - py]
        # return [self.ppx, self.ppy, self.pvx, self.pvy, self.tx - self.ppx, self.ty - self.ppy,
        #         self.tx2 - self.ppx, self.ty2 - self.ppy]
    #

    def step(self, action):
        self.prev_ppx = self.ppx
        self.prev_ppy = self.ppy

        speed = np.sqrt(self.pvx**2 + self.pvy*2)
        if speed > self.max_velocity:
            self.pvx *= self.max_velocity/speed
            self.pvy *= self.max_velocity/speed

        self.ppx += self.pvx/self.update_rate
        self.ppy += self.pvy/self.update_rate

        self.pvx *= self.damping**self.damping_per_step
        self.pvy *= self.damping**self.damping_per_step

        if action < len(self.actions):
            a = self.actions[action]
            if action < self.x_actions_len:
                if a < self.pvx:
                    self.pvx -= self.access_per_step
                elif a > self.pvx:
                    self.pvx += self.access_per_step
            if action >= self.x_actions_len or a == 0.0:
                if a < self.pvy:
                    self.pvy -= self.access_per_step
                elif a > self.pvy:
                    self.pvy += self.access_per_step
        # if action == 0:
        #     self.pvx -= self.access_per_step
        # if action == 1:
        #     self.pvx += self.access_per_step
        # if action == 2:
        #     self.pvy -= self.access_per_step
        # if action == 3:
        #     self.pvy += self.access_per_step


        if self.ppx < self.x_start + self.drone_radius:
            self.pvx *= -0.5
            self.ppx = self.x_start + self.drone_radius + 0.02

        if self.ppx > self.x_end - self.drone_radius:
            self.pvx *= -0.5
            self.ppx = self.x_end - self.drone_radius - 0.02

        if self.ppy < self.y_start + self.drone_radius:
            self.pvy *= -0.5
            self.ppy = self.y_start + self.drone_radius + 0.02

        if self.ppy > self.y_end - self.drone_radius:
            self.pvy *= -0.5
            self.ppy = self.y_end - self.drone_radius - 0.02

        self.t += 1
        if self.t % self.update_goal_step == 0:
            self.tx = np.random.uniform(self.x_start + self.drone_radius, self.x_end-self.drone_radius)
            self.ty = np.random.uniform(self.y_start + self.drone_radius, self.y_end-self.drone_radius)


        #compute distances
        dx = self.ppx - self.tx
        dy = self.ppy - self.ty
        d1 = np.sqrt(dx * dx+dy * dy)

        dx = self.ppx - self.tx2
        dy = self.ppy - self.ty2
        d2 = np.sqrt(dx * dx+dy * dy)

        dxnorm = dx / d2
        dynorm = dy / d2
        self.tx2 += self.bad_distance_step * dxnorm
        self.ty2 += self.bad_distance_step * dynorm

        # done = (d2 <= self.drone_radius)
        done = False
        r = -0.5
        if action < len(self.actions):
            a = self.actions[action]
            if abs(dx) > abs(dy):
                if action < self.x_actions_len:
                    if (self.ppx - self.tx < -0.1) and a > 0.0:
                        r = 0.5
                    if (self.ppx - self.tx > 0.1) and a < 0.0:
                        r = 0.5
            else:
                if action >= self.x_actions_len:
                    if (self.ppy - self.ty < -0.1) and a > 0.0:
                        r = 0.5
                        # print "3"
                    if (self.ppy - self.ty > 0.1) and a < 0.0:
                        r = 0.5
                # print "2"
        # if (-0.1 <= dx <= 0.1) and action == 4:
        #     r = 8.0
        # if (-0.1 <= dy <= 0.1) and action == 4:
        #     r = 8.0
        # r += 4.0 - ((abs(dx) / self.wall_size_x) * 8.0) + 4.0 - ((abs(dy) / self.wall_size_y) * 8.0)
        # r = (1.0 - (d1 / self.max_distance)) * 2.0
        # r += self.max_distance - d1
        # print r , d1
        # r = 4.0 - ((abs(dx) / self.wall_size_x) * 8.0)
        # d2 /= self.max_distance
        bad = self.bad_radius / self.max_distance
        # r = -(d1 / self.max_distance)*2
        # r = -d1/self.max_distance
        d1 = self.max_distance if d1 > self.max_distance else d1
        r = -(d1/self.max_distance)
        # print r
        #compute reward
        # r = 1.0 - d1**2
        # print r
        # if (d2/self.max_distance) < bad:
        # if d2 < self.bad_radius:
        #     r += 2 * (d2 - self.bad_radius) / self.bad_radius
        #     pass
        # if action == 2:
        #     r -= 2.0
        ns = self.get_state()
        # print r
        # print action, r

        return ns, r, done

    def render(self):
        if self.show:
            self.frontend.render()