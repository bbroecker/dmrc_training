import yaml

import numpy as np

from tf_nn_sim.Environments import Environment
from tf_nn_sim.Environments import PuckNormalMultiFrontend

CONFIG_FILE = "../Config/PuckNormalMulti/puckworld.yaml"

MIN_OUTPUT = -5.0
MAX_OUTPUT = 5.0

# MIN_OUTPUT = -1.0
# MAX_OUTPUT = 1.0
REWARD_OUTPUT_MIN = -1.0
REWARD_OUTPUT_MAX = 1.0

class PuckNormalMultiworld(Environment):
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
        self.bad_speed = None
        self.meter_x = None
        self.area_pixel_x = None
        self.debug_pixel_x = None
        self.area_pixel_y = None
        self.debug_pixel_y = None
        self.meter_y = None
        self.x_start = None
        self.x_end = None
        self.y_start = None
        self.num_drones = None
        self.y_end = None
        self.max_goal_distance = None
        self.load_cfg(CONFIG_FILE)
        self.damping_per_step = self.update_rate/self.damping_rate
        self.reset()
        self.num_states = len(self.get_state(0))
        self.access_per_step = self.access * 1.0 / self.update_rate
        self.update_goal_step = self.goal_update_steps
        self.bad_distance_step = self.bad_speed * 1.0/self.update_rate
        self.show = show
        self.goal_count = 0
        if show:
            self.frontend = PuckNormalMultiFrontend(self)


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
            self.pvx[i] = np.random.uniform(self.x_start, self.x_end) * 0.05 - 0.025
            self.pvy[i] = np.random.uniform(self.y_start, self.y_end) * 0.05 - 0.025
            self.tx[i] = np.random.uniform(0.0, 1.0)
            self.ty[i] = np.random.uniform(0.0, 1.0)

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
        self.damping = cfg['dynamics']['damping']
        self.damping_rate = cfg['dynamics']['damping_rate']
        self.bad_radius = cfg['world']['bad_radius']
        self.drone_radius = cfg['world']['drone_radius']
        self.goal_update_steps = cfg['world']['goal_update_steps']
        self.bad_speed = cfg['world']['bad_speed']
        self.max_goal_distance = cfg['world']['max_goal_distance']
        self.num_drones = cfg['world']['num_drones']
        actions = cfg['controlls']['actions']
        self.update_rate = cfg['controlls']['update_rate']
        self.area_pixel_x = cfg['frontend']['area_pixel_x']
        self.debug_pixel_x = cfg['frontend']['debug_pixel_x']
        self.area_pixel_y = cfg['frontend']['area_pixel_y']
        self.debug_pixel_y = cfg['frontend']['debug_pixel_y']
        self.x_actions_len = len(actions)
        self.actions = self.generate_action(actions)
        # self.num_actions = len(self.actions) + 1
        self.ppx = [0.0] * self.num_drones
        self.prev_ppx = [0.0] * self.num_drones
        self.ppy = [0.0] * self.num_drones
        self.prev_ppy = [0.0] * self.num_drones
        self.pvx = [0.0] * self.num_drones
        self.pvy = [0.0] * self.num_drones
        self.tx = [0.0] * self.num_drones
        self.ty = [0.0] * self.num_drones
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
        return angle/np.pi

    def get_angle_state(self, source_id, other_id):
        px = self.ppx[source_id]
        py = self.ppy[source_id]
        pvx = self.pvx[source_id]
        pvx_other = self.pvx[other_id]
        pvy = self.pvy[source_id]
        pvy_other = self.pvy[other_id]
        tx = self.ppx[other_id]
        ty = self.ppy[other_id]
        # return [(tx - px), ty - py]
        dx = tx - px
        dy = ty - py

        d1 = np.sqrt(dx * dx+dy * dy) #target

        # return [dx / d1, dy / d1]
        return [pvx, pvy, pvx_other, pvy_other, d1]

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
        tx = self.tx[drone_id]
        ty = self.ty[drone_id]
        if c_id is not None:
            tx2 = self.ppx[c_id]
            ty2 = self.ppy[c_id]
            # return [(tx - px), ty - py]
            dx = tx - px
            dy = ty - py
            dx2 = self.ppx[drone_id] - tx2
            dy2 = self.ppy[drone_id] - ty2
            pvx_o = self.pvx[c_id]
            pvy_o = self.pvy[c_id]
            px = self.translate(px, 0.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            py = self.translate(py, 0.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            pvx = self.translate(pvx*100.0, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            pvy = self.translate(pvy*100.0, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            dx = self.translate(dx, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            dy = self.translate(dy, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            dx2 = self.translate(dx2, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            dy2 = self.translate(dy2, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)

            d1 = np.sqrt(dx * dx+dy * dy) #target
            d2 = np.sqrt(dx2 * dx2+dy2 * dy2) #bad

            # return [pvx, pvy, d1/self.max_distance, d2/self.max_distance]
            # return [px, py, pvx, pvy, tx - px, ty - py, tx2 - px, ty2 - py]
            # return [px -0.5, py -0.5, pvx*10.0, pvy*10.0, tx - px, ty - py, tx2 - px, ty2 - py]
            # return [px -0.5, py -0.5, pvx*10.0, pvy*10.0, pvx_o * 10.0 ,pvy_o * 10.0, tx - px, ty - py, tx2 - px, ty2 - py]
            return [px, py, pvx, pvy, dx, dy, dx2, dy2]
            # return [pvx*10.0, pvy*10.0, tx - px, ty - py, tx2 - px, ty2 - py]
        else:
            # return [px - 0.5, py - 0.5, pvx * 10.0, pvy * 10.0, tx - px, ty - py]
            dx = tx - px
            dy = ty - py
            px = self.translate(px, 0.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            py = self.translate(py, 0.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            # print pvx, pvy
            pvx = self.translate(pvx*100.0, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            pvy = self.translate(pvy*100.0, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            dx = self.translate(dx, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            dy = self.translate(dy, -1.0, 1.0, MIN_OUTPUT, MAX_OUTPUT)
            # state = [px, py, pvx, pvy, dx, dy]
            state = [pvx, pvy, dx, dy]
            # state = [px, py, pvx, pvy, dx, dy]
            # print "ty {0} py {1} state {2}".format(ty, py, state)
            return state




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

        d2 = np.sqrt(dx2 * dx2+dy2 * dy2) #bad
        # return [self.tx2 - self.ppx, self.ty2 - self.ppy]
        return [pvx, pvy, d2, 0.95]
        # return [px, py, pvx, pvy, d2, 0.95]
        # return [px, py, d2]

    def single_step(self, id, action):
        self.prev_ppx[id] = self.ppx[id]
        self.prev_ppy[id] = self.ppy[id]
        self.ppx[id] += self.pvx[id]
        self.ppy[id] += self.pvy[id]

        self.pvx[id] *= 0.95
        self.pvy[id] *= 0.95

        # accel = 0.004
        accel = 0.001
        # accel = 0.004
        if action == 0:
            self.pvx[id] -= accel
        if action == 1:
            self.pvx[id] += accel
        if action == 2:
            self.pvy[id] -= accel
        if action == 3:
            self.pvy[id] += accel
        # if action == 4:
        #     self.pvy[id] += accel
        #     self.pvx[id] += accel
        # if action == 5:
        #     self.pvy[id] -= accel
        #     self.pvx[id] -= accel
        # if action == 6:
        #     self.pvy[id] += accel
        #     self.pvx[id] -= accel
        # if action == 7:
        #     self.pvy[id] -= accel
        #     self.pvx[id] += accel


        if self.ppx[id] < self.drone_radius:
            self.pvx[id] *= -0.5
            self.ppx[id] = self.drone_radius

        if self.ppx[id] > 1.0 - self.drone_radius:
            self.pvx[id] *= -0.5
            self.ppx[id] = 1.0 - self.drone_radius

        if self.ppy[id] < self.drone_radius:
            self.pvy[id] *= -0.5
            self.ppy[id] = self.drone_radius

        if self.ppy[id] > 1.0 - self.drone_radius:
            self.pvy[id] *= -0.5
            self.ppy[id] = 1.0 - self.drone_radius

        dx = self.ppx[id] - self.tx[id]
        dy = self.ppy[id] - self.ty[id]
        d1 = np.sqrt(dx * dx+dy * dy)



    def calc_reward(self, actions, id):
        done = False
        #compute distances
        dx = self.ppx[id] - self.tx[id]
        dy = self.ppy[id] - self.ty[id]
        d1 = np.sqrt(dx * dx+dy * dy)
        # d1 = abs(dx) + abs(dy)

        # if d1 <self.bad_radius:
        #     r = -(d1**2)
        # else:
        r = -d1
        # r = self.translate(r, -1.0, 0.0, -1.0, 1.0)

        # print " distance {0} reward {1}".format(-d1, r)
        # print "reward {0}".format(r)
        # if d1 >= 0.05 and actions[id] == self.num_actions - 1:
        #     print "close enough"
            # r = - 0.2
        d2 = 1000000.0
        if self.num_drones > 1:
            c_id = self.closes_drone(id)

            if c_id is not None:
                dx = self.ppx[id] - self.ppx[c_id]
                dy = self.ppy[id] - self.ppy[c_id]
                d2 = np.sqrt(dx * dx+dy * dy)

                if d2 < self.bad_radius:
                    r += 2 * (d2 - self.bad_radius) / self.bad_radius
                    # if d2 < 0.1:
                    #     r = -5.0
                    #     done = True

            if self.t % self.update_goal_step == 0 or d1 <= 0.1:
            # if self.t % self.update_goal_step == 0:
                self.tx[id] = np.random.uniform(0, 1.0)
                self.ty[id] = np.random.uniform(0.0, 1.0)
                # if d1 <= 0.1:
                #     self.goal_count += 1

            r = self.translate(r, -3.0, 0.0, REWARD_OUTPUT_MIN, REWARD_OUTPUT_MAX)
        else:
            r = self.translate(r, -1.0, 0.0, REWARD_OUTPUT_MIN, REWARD_OUTPUT_MAX)
        if d1 <= 0.025 or d2 <= 0.025:
            done = True

        return r, done


    def step(self, actions):
        if len(actions) != self.num_drones:
            pass
        self.t += 1

        for i in range(self.num_drones):
            self.single_step(i, actions[i])

        rewards = [0] * self.num_drones
        new_states = [0] * self.num_drones
        done = [False] * self.num_drones

        for i in range(self.num_drones):
            rewards[i], done[i] = self.calc_reward(actions, i)
            new_states[i] = self.get_state(i)

        # done = False

        return new_states, rewards, done

    def render(self, debug_data):
        if self.show:
            self.frontend.render(debug_data)