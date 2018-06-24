import yaml

import numpy as np
from teacher.Astar import AStar

from tf_nn_sim.Environments import Environment
from tf_nn_sim.Environments import PuckGridFrontend

CONFIG_FILE = "Config/PuckGrid/puckworld.yaml"


class PuckGrid(Environment):
    def __init__(self, show=True):

        Environment.__init__(self, CONFIG_FILE)
        self.critical_distance = None
        self.grid_size_x = None
        self.grid_size_y = None
        self.num_actions = None
        self.drone_radius = None
        self.ppx = None
        self.prev_ppx = None
        self.ppy = None
        self.prev_ppy = None
        self.tx = None
        self.ty = None
        self.ox = None
        self.oy = None
        self.t = None
        self.goal_update_rate = None
        self.bad_update_rate = None
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.y_end = None
        self.pixel_x = None
        self.pixel_y = None
        self.max_goal_distance = None
        self.step_skip = None
        self.load_cfg(CONFIG_FILE)
        self.max_distance = float(np.sqrt(self.grid_size_x ** 2 + self.grid_size_y ** 2))
        self.x_start = 0
        self.x_end = self.grid_size_x - 1
        self.y_start = 0
        self.y_end = self.grid_size_y - 1
        self.ppx = np.random.randint(self.x_start, self.x_end)
        self.ppy = np.random.randint(self.y_start, self.y_end)
        self.prev_ppx = self.ppx
        self.prev_ppy = self.ppy
        self.reset()
        self.num_states = len(self.get_state())
        self.show = show
        if show:
            self.frontend = PuckGridFrontend(self)

    def reset(self):

        # self.ppx = np.random.random() * self.wall_size_x
        # self.ppx = np.random.randint(self.x_start, self.x_end)
        # self.ppx = 0
        # self.ppy = np.random.randint(self.y_start, self.y_end)
        # self.ppy = self.y_end
        # self.prev_ppx = self.ppx
        # self.prev_ppy = self.ppy
        self.tx = np.random.randint(self.x_start, self.x_end)
        self.ty = np.random.randint(self.y_start, self.y_end)
        self.ox = np.random.randint(self.x_start, self.x_end)
        self.oy = np.random.randint(self.y_start, self.y_end)

        while self.tx == self.ppx and self.ty == self.ppy:
            self.tx = np.random.randint(self.x_start, self.x_end)
            self.ty = np.random.randint(self.y_start, self.y_end)
        while (self.tx == self.ox and self.ty == self.oy) or (self.ppx == self.ox and self.ppy == self.oy):
            self.ox = np.random.randint(self.x_start, self.x_end)
            self.oy = np.random.randint(self.y_start, self.y_end)

        self.t = 0
        return self.get_state()

    def load_cfg(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                cfg = yaml.load(stream)['Environment']
            except yaml.YAMLError as exc:
                print(exc)

        self.drone_radius = cfg['world']['drone_radius']
        self.grid_size_x = cfg['world']['grid_size_x']
        self.grid_size_y = cfg['world']['grid_size_y']
        self.goal_update_rate = cfg['world']['goal_update_rate']
        self.bad_update_rate = cfg['world']['bad_update_rate']
        self.critical_distance = cfg['world']['bad_radius']
        self.max_goal_distance = cfg['world']['max_goal_distance']
        self.pixel_x = cfg['frontend']['pixel_x']
        self.pixel_y = cfg['frontend']['pixel_y']
        self.step_skip = cfg['controlls']['step_skip']
        self.num_actions = 5
        # self.num_actions = 3

    def get_num_actions(self):
        return self.num_actions

    def get_num_states(self):
        return self.num_states

    def get_state(self):

        pvx = self.ppx - self.prev_ppx
        pvy = self.ppy - self.prev_ppy

        # return [(tx - px), ty - py]
        dx1 = self.tx - self.ppx
        dy1 = self.ty - self.ppy
        d1 = np.sqrt(dx1 * dx1 + dy1 * dy1)  # target
        # d1 = abs(dx) + abs(dy) #manhatten

        dx2 = self.ox - self.ppx
        dy2 = self.oy - self.ppy
        d2 = np.sqrt(dx2 * dx2 + dy2 * dy2)  # bad
        # d2 = abs(dx) + abs(dy)  # manhatten

        # return [pvx / self.max_distance, pvy / self.max_distance, d1/self.max_distance, d2/self.max_distance]
        # print [self.ppx / self.max_distance, self.ppy / self.max_distance, dx1 / self.max_distance, dy1 / self.max_distance,
        #  dx2 / self.max_distance, dy2 / self.max_distance]
        return [self.ppx / self.max_distance, self.ppy / self.max_distance, pvx / self.max_distance,
                pvy / self.max_distance, dx1 / self.max_distance, dy1 / self.max_distance, dx2 / self.max_distance,
                dy2 / self.max_distance]
        # return [self.ppx, self.ppy, pvx, pvy, dx1, dy1, dx2, dy2]
        # return [self.ppx, self.ppy, pvx, pvy, dx1, dy1, dx2, dy2]

    def astar_reward(self, action):
        reward = -.2




        if self.ppx != self.tx or self.ppy != self.ty:
            astar = AStar()
            astar.init_grid(self.grid_size_x, self.grid_size_y, [(self.ox, self.oy)], (self.ppx, self.ppy), (self.tx, self.ty))
            l = astar.solve()

            if l is not None:
                p = l[1]
                x, y = p[0], p[1]
                if x != self.ppx:
                    if x < self.ppx and action == 0:
                        reward = .2
                    elif action == 1:
                        reward = .2
                else:
                    if y < self.ppy and action == 2:
                        reward = .2
                    elif action == 3:
                        reward = .2
        return reward


    def step(self, action):
        r = 0
        self.prev_ppx = self.ppx
        self.prev_ppy = self.ppy
        for s in range(0, self.step_skip +1):

            # r += self.astar_reward(action)

            if action == 0:
                self.ppx -= 1
            elif action == 1:
                self.ppx += 1
            elif action == 2:
                self.ppy -= 1
            elif action == 3:
                self.ppy += 1

            if self.ppx < self.x_start:
                self.ppx = self.x_start
            if self.ppx > self.x_end:
                self.ppx = self.x_end

            if self.ppy < self.y_start:
                self.ppy = self.y_start

            if self.ppy > self.y_end:
                self.ppy = self.y_end

            dx = self.ppx - self.tx
            dy = self.ppy - self.ty
            d1 = np.sqrt(dx * dx + dy * dy)  # target
            # d1 = abs(dx) + abs(dy) #manhatten

            dx = self.ppx - self.ox
            dy = self.ppy - self.oy
            d2 = np.sqrt(dx * dx + dy * dy)  # bad

            self.t += 1
            # if self.t % self.goal_update_rate == 0 or d1 == 0:
            if int(d1) <= 1:
                print "goal!!"
                r += 0.25
                self.tx = np.random.randint(self.x_start, self.x_end)
                self.ty = np.random.randint(self.y_start, self.y_end)
                while self.tx == self.ox and self.ty == self.oy:
                    self.tx = np.random.randint(self.x_start, self.x_end)
                    self.ty = np.random.randint(self.y_start, self.y_end)

            done = False

            # d2 = abs(dx) + abs(dy)  # manhatten

            r += -int(d1) / self.max_distance
            if self.prev_ppx != self.ppx or self.prev_ppy != self.ppy:
                r += 0.2
            # print d2
            if d2 <= self.critical_distance:
                # print "now ", d2
                r += 2 * (int(d2) - self.critical_distance) / (self.critical_distance)
            # print r
            # if d1 == 0:
            #     r += 0.5
            if d2 == 0:
                r = -3.0
                done = True
                break

        ns = self.get_state()
        # print r
        # print action, r

        return ns, r, done

    def render(self):
        if self.show:
            self.frontend.render()
