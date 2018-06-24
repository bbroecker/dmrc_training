import yaml

import numpy as np
from Environments.Environment import Environment

from tf_nn_sim.Environments.PuckNormal import PuckNormalFrontend

CONFIG_FILE = "../Config/PuckNormal/puckworld.yaml"


class PuckNormalworld(Environment):
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
        self.tx2 = None
        self.ty2 = None
        self.t = None
        self.update_rate = None
        self.angle_obstacle = None
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
        self.damping_per_step = self.update_rate/self.damping_rate
        self.reset()
        self.num_states = len(self.get_state())
        self.access_per_step = self.access * 1.0 / self.update_rate
        self.update_goal_step = self.goal_update_steps
        self.bad_distance_step = self.bad_speed * 1.0/self.update_rate
        self.show = show
        if show:
            self.frontend = PuckNormalFrontend(self)


    def reset(self):
        self.x_start = 0.0
        self.x_end = 0.0
        self.y_start = 1.0
        self.y_end = 1.0
        # self.ppx = np.random.random() * self.wall_size_x
        self.ppx = np.random.uniform(self.x_start, self.x_end)
        self.prev_ppx = self.ppx
        self.ppy = np.random.uniform(self.y_start, self.y_end)
        self.prev_ppy = self.ppy
        self.pvx = np.random.uniform(self.x_start, self.x_end) * 0.05 -0.025
        self.pvy = np.random.uniform(self.x_start, self.x_end) * 0.05 -0.025
        self.tx = np.random.uniform(0.0, 1.0)
        self.ty = np.random.uniform(0.0, 1.0)
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
        self.goal_update_steps = cfg['world']['goal_update_steps']
        self.bad_speed = cfg['world']['bad_speed']
        self.max_goal_distance = cfg['world']['max_goal_distance']
        actions = cfg['controlls']['actions']
        self.update_rate = cfg['controlls']['update_rate']
        self.pixel_x = cfg['frontend']['pixel_x']
        self.pixel_y = cfg['frontend']['pixel_y']
        self.x_actions_len = len(actions)
        self.actions = self.generate_action(actions)
        # self.num_actions = len(self.actions) + 1
        self.num_actions = 5

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

    def get_angle_target(self):
        tx = self.tx - self.ppx
        ty = self.ty - self.ppy
        angle = np.math.atan2(ty, tx)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle/np.pi

    def get_angle_obstacle(self):
        tx = self.tx2 - self.ppx
        ty = self.ty2 - self.ppy
        angle = np.math.atan2(ty, tx)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle/np.pi

    def get_state(self):
        px = self.ppx
        py = self.ppy
        pvx = self.pvx
        pvy = self.pvy
        tx = self.tx
        ty = self.ty
        tx2 = self.tx2
        ty2 = self.ty2
        # return [(tx - px), ty - py]
        dx = self.ppx - self.tx
        dy = self.ppy - self.ty
        dx2 = self.ppx - self.tx2
        dy2 = self.ppy - self.ty2
        d1 = np.sqrt(dx * dx+dy * dy) #target
        d2 = np.sqrt(dx2 * dx2+dy2 * dy2) #bad

        # return [pvx, pvy, d1/self.max_distance, d2/self.max_distance]
        # return [px, py, pvx, pvy, tx - px, ty - py, tx2 - px, ty2 - py]
        return [px -0.5, py -0.5, pvx*10.0, pvy*10.0, tx - px, ty - py, tx2 - px, ty2 - py]
        # return [pvx * 10.0, pvy * 10.0, d1, d2]
        # return [pvx * 10.0, pvy * 10.0, d1]
        # return [self.ppx, self.ppy, self.pvx, self.pvy, self.tx - self.ppx, self.ty - self.ppy,
        #         self.tx2 - self.ppx, self.ty2 - self.ppy]
    #

    def get_target_state(self):
        px = self.ppx
        py = self.ppy
        pvx = self.pvx
        pvy = self.pvy
        tx = self.tx
        ty = self.ty
        tx2 = self.tx2
        ty2 = self.ty2
        # return [(tx - px), ty - py]
        dx = self.ppx - self.tx
        dy = self.ppy - self.ty

        d1 = np.sqrt(dx * dx+dy * dy) #target

        return [self.tx - self.ppx, self.ty - self.ppy]
        # return [pvx, pvy, d1]
        # return [px, py, d1]

    def get_obstacle_state(self):
        px = self.ppx
        py = self.ppy
        pvx = self.pvx
        pvy = self.pvy
        tx2 = self.tx2
        ty2 = self.ty2

        dx2 = self.ppx - self.tx2
        dy2 = self.ppy - self.ty2

        d2 = np.sqrt(dx2 * dx2+dy2 * dy2) #bad
        # return [self.tx2 - self.ppx, self.ty2 - self.ppy]
        return [pvx, pvy, d2, 0.95]
        # return [px, py, pvx, pvy, d2, 0.95]
        # return [px, py, d2]

    def step(self, action):
        self.prev_ppx = self.ppx
        self.prev_ppy = self.ppy
        self.ppx += self.pvx
        self.ppy += self.pvy

        self.pvx *= 0.95
        self.pvy *= 0.95

        # accel = 0.002
        accel = 0.004
        if action == 0:
            self.pvx -= accel
        if action == 1:
            self.pvx += accel
        if action == 2:
            self.pvy -= accel
        if action == 3:
            self.pvy += accel


        if self.ppx < self.drone_radius:
            self.pvx *= -0.5
            self.ppx = self.drone_radius

        if self.ppx > 1.0 - self.drone_radius:
            self.pvx *= -0.5
            self.ppx = 1.0 - self.drone_radius

        if self.ppy < self.drone_radius:
            self.pvy *= -0.5
            self.ppy = self.drone_radius

        if self.ppy > 1.0 - self.drone_radius:
            self.pvy *= -0.5
            self.ppy = 1.0 - self.drone_radius

        self.t += 1
        if self.t % self.update_goal_step == 0:
            self.tx = np.random.uniform(0, 1.0)
            self.ty = np.random.uniform(0.0, 1.0)


        #compute distances
        dx = self.ppx - self.tx
        dy = self.ppy - self.ty
        d1 = np.sqrt(dx * dx+dy * dy)

        dx = self.ppx - self.tx2
        dy = self.ppy - self.ty2
        d2 = np.sqrt(dx * dx+dy * dy)

        dxnorm = dx / d2
        dynorm = dy / d2
        speed = 0.001
        self.tx2 += speed * dxnorm
        self.ty2 += speed * dynorm

        # done = (d2 <= self.drone_radius)
        done = False

        r = -d1

        if d2 < self.bad_radius:
            r += 2 * (d2 - self.bad_radius) / self.bad_radius

        ns = self.get_state()

        return ns, r, done

    def render(self):
        if self.show:
            self.frontend.render()