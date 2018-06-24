import threading

import numpy as np
import copy

# from tf_nn_sim.v2.helper import wrap_angle
import scipy.stats

from v2.helper import wrap_angle


class ParticleFilter2_5D_Cfg(object):
    def __init__(self):
        self.sample_size = 100
        self.x_limit = [-2.0, 2.0]
        self.resample_x_limit = [-0.01, 0.01]
        self.y_limit = [-2.0, 2.0]
        self.resample_y_limit = [-0.01, 0.01]
        self.yaw_limit = [-np.pi, np.pi]
        self.resample_yaw_limit = [-np.pi*0.01, np.pi*0.01]
        self.respawn_yaw_limit = [-np.pi*0.1, np.pi*0.1]
        self.respawn_y_limit = [-0.02, 0.02]
        self.respawn_x_limit = [-0.02, 0.02]
        self.new_spawn_samples = 0.1
        self.mean = 0.0
        self.variance = 0.0312979988458
        self.resample_prob = 1.0

class Particle:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.weight = 0.0

def rotate_vel(x_vel, y_vel, angle):
    x = x_vel * np.cos(angle) - y_vel * np.sin(angle)
    y = x_vel * np.sin(angle) + y_vel * np.cos(angle)
    return x, y

def probability(measure_error, cfg):
    assert isinstance(cfg, ParticleFilter2_5D_Cfg)
    p = (1.0 / (np.sqrt(2.0 * np.pi * cfg.variance))) * np.exp(-(((measure_error - cfg.mean) ** 2)/(2.0*cfg.variance)))
    # p = scipy.stats.norm(cfg.mean, np.sqrt(cfg.variance)).cdf(measure_error)
    # print p
    return p

class ParticleFilterNN:
    def __init__(self, cfg):
        # assert isinstance(cfg, ParticleFilter2_5D_Cfg)
        self.cfg = cfg
        self.samples = []
        self.normalise_factor = 0.0
        self.call_counter = 0
        self.predict_position = Particle(0.0, 0.0, 0.0)
        self.samples_lock = threading.Semaphore()
        self.max_weight = 0.0
        # print "constructor reset"
        self.reset()


    def reset(self, guess_x=None, guess_y=None, guess_yaw=None):
        # print "reset filter"
        self.samples_lock.acquire()
        self.samples = []
        self.call_counter = 0
        self.normalise_factor = 0.0
        self.predict_position = Particle(0.0, 0.0, 0.0)
        for i in range(self.cfg.sample_size):
            if guess_x is None:
                x = np.random.uniform(self.cfg.x_limit[0], self.cfg.x_limit[1])
            else:
                x = guess_x + np.random.uniform(self.cfg.respawn_x_limit[0], self.cfg.respawn_x_limit[1])
            if guess_y is None:
                y = np.random.uniform(self.cfg.y_limit[0], self.cfg.y_limit[1])
            else:
                y = guess_y + np.random.uniform(self.cfg.respawn_y_limit[0], self.cfg.respawn_y_limit[1])
            if guess_yaw is None:
                yaw = np.random.uniform(self.cfg.yaw_limit[0], self.cfg.yaw_limit[1])
            else:
                # print "guess yaw!!"
                yaw = wrap_angle(guess_yaw + np.random.uniform(self.cfg.respawn_yaw_limit[0], self.cfg.respawn_yaw_limit[1]))

            # print "x {} y {} i {}".format(x, y, i)
            self.samples.append(Particle(x, y, yaw))
        self.samples_lock.release()


    def update_samples(self, vx, vy, o_vx, o_vy, dt, distance_measure, guess_x, guess_y, guess_yaw):
        # print vx, vy, o_vx, o_vy, dt, distance_measure, guess_x, guess_y, guess_yaw
        # d = np.sqrt((vx * dt)**2 + (vy * dt)**2)
        # print "particle distance: {}".format(d)
        self.samples_lock.acquire()
        self.motion_update(vx, vy, o_vx, o_vy, dt)
        self.sensor_update(distance_measure)
        self.low_variance_resampling(distance_measure, guess_x, guess_y, guess_yaw)
        self.samples_lock.release()

    # def rotate_vel(x_vel, y_vel, angle):
    #     x = x_vel * np.cos(angle) - y_vel * np.sin(angle)
    #     y = x_vel * np.sin(angle) + y_vel * np.cos(angle)
    #     return x, y

    def motion_update(self, vx, vy, o_vx, o_vy, dt):
        for s in self.samples:
            # pass
            s.x += -vx * dt
            s.y += -vy * dt
            tmp_x, tmp_y = rotate_vel(o_vx, o_vy, s.yaw)
            s.x += tmp_x * dt
            s.y += tmp_y * dt

    def calc_weight(self, x, y, distance_measure):
        distance_error = np.sqrt(x ** 2 + y ** 2) - distance_measure
        p = probability(distance_error, self.cfg)
        return p

    def sensor_update(self, distance_measure):
        self.max_weight = 0.0
        m = probability(0.0, self.cfg)
        predict_x = self.predict_position.x
        predict_y = self.predict_position.y
        max_distance = 0.0
        min_distance = 10000.0
        w = 0.0
        index = 0
        self.predict_position.x = self.predict_position.y = self.predict_position.yaw = 0.0
        self.normalise_factor = 0.0
        # average pos
        for idx, s in enumerate(self.samples):
            p = self.calc_weight(s.x, s.y, distance_measure) / m
            d = np.sqrt((predict_x-s.x)** 2 + (predict_y-s.y)** 2)
            min_distance = min(d, min_distance)
            if d > max_distance:
                index = idx
                max_distance = d
                w = s.weight
            self.max_weight = max(self.max_weight, p)
            # print distance_error, p
            s.weight = p
            self.normalise_factor += p
            self.predict_position.x += (s.x * s.weight)
            self.predict_position.y += (s.y * s.weight)
            self.predict_position.yaw += (s.yaw * s.weight)
        if self.normalise_factor <= 0.0:
            pass
        # self.normalise_factor = 1.0 / len(self.samples * 10) if self.normalise_factor <= 0 else self.normalise_factor
        else:
            self.predict_position.x /= self.normalise_factor
            self.predict_position.y /= self.normalise_factor
            self.predict_position.yaw /= self.normalise_factor
            self.predict_position.yaw = wrap_angle(self.predict_position.yaw)

        # heighest weight
        # for idx, s in enumerate(self.samples):
        #     p = self.calc_weight(s.x, s.y, distance_measure) / m
        #     d = np.sqrt((predict_x-s.x)** 2 + (predict_y-s.y)** 2)
        #     min_distance = min(d, min_distance)
        #     if d > max_distance:
        #         index = idx
        #         max_distance = d
        #         w = s.weight
        #     if p > self.max_weight:
        #         self.max_weight = p
        #         self.predict_position.x = s.x
        #         self.predict_position.y = s.y
        #         self.predict_position.yaw = s.yaw
        #
        #
        #     # print distance_error, p
        #     s.weight = p
        #     self.normalise_factor += p



        # if max_distance > 1.0:
        #     print max_distance, w, index


    def low_variance_resampling(self, distance_measure, guess_x, guess_y, guess_yaw):
        if self.normalise_factor == 0:
            return
        new_samples = []
        if self.call_counter % 1 == 0 and self.max_weight < self.cfg.resample_prob:
            new_spawn_size = int(round(self.cfg.sample_size * self.cfg.new_spawn_samples))
            # print "respawn", new_spawn_size
            resample_size = self.cfg.sample_size - new_spawn_size
        else:
            new_spawn_size = 0
            resample_size = self.cfg.sample_size
        r = np.random.uniform(0.0, 1.0 / resample_size)
        c = self.samples[0].weight / self.normalise_factor
        # print r, c
        i = 0

        for m in range(resample_size):
            u = r + float(m) / resample_size
            if u < 0.0:
                print "can't be right"
            # u = r
            # print "u: {}".format(u)
            while u > c:
                i += 1
                i = (len(self.samples) - 1) if i >= len(self.samples) else i
                c += self.samples[i].weight/self.normalise_factor
                # print "sampling: {}".format(self.samples[i].weight)
                # print "sampling: {}".format(self.samples[i].weight/self.normalise_factor)
            # c = self.samples[i].weight/self.normalise_factor
            # print "add", self.samples[i].x, self.samples[i].y,i
            n_s = copy.copy(self.samples[i])
            random_factor = 1.0 - n_s.weight
            random_factor = 0.0 if random_factor < 0.0 else random_factor
            # print n_s.weight, random_factor
            n_s.x += np.random.uniform(self.cfg.resample_x_limit[0] * random_factor, self.cfg.resample_x_limit[1] * random_factor)
            n_s.y += np.random.uniform(self.cfg.resample_y_limit[0] * random_factor, self.cfg.resample_y_limit[1] * random_factor)
            n_s.yaw += np.random.uniform(self.cfg.resample_yaw_limit[0] * random_factor, self.cfg.resample_yaw_limit[1] * random_factor)
            n_s.yaw = wrap_angle(n_s.yaw)
            n_s.weight = self.calc_weight(n_s.x, n_s.y, distance_measure)
            new_samples.append(n_s)

        # if new_spawn_size > 0:
        #     print "resample!!", self.call_counter

        for m in range(new_spawn_size):
            x = guess_x + np.random.uniform(self.cfg.respawn_x_limit[0], self.cfg.respawn_x_limit[1])
            y = guess_y + np.random.uniform(self.cfg.respawn_y_limit[0], self.cfg.respawn_y_limit[1])
            yaw = wrap_angle(guess_yaw + np.random.uniform(self.cfg.respawn_yaw_limit[0], self.cfg.respawn_yaw_limit[1]))
            s = Particle(x, y, yaw)
            s.weight = self.calc_weight(x, y, distance_measure)
            new_samples.append(s)

        self.samples = new_samples
        self.call_counter += 1

    def get_angle(self):
        return np.arctan2(self.predict_position.y, self.predict_position.x)

    def global_pos(self, x, y, orientation):
        p_x, p_y = rotate_vel(self.predict_position.x, self.predict_position.y, orientation)
        return x + p_x, y + p_y

    def global_pose(self, x, y, orientation):
        p_x, p_y = rotate_vel(self.predict_position.x, self.predict_position.y, orientation)
        return x + p_x, y + p_y, wrap_angle(self.predict_position.yaw + orientation)

    def local_pose(self):
        return self.predict_position.x, self.predict_position.y, self.predict_position.yaw

