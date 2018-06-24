# Define the colors we will use in RGB format
import pygame
import sys

import numpy as np

# import tf_nn_sim.Environments.PuckRealMulti.Puckworld_Environment.PuckRealMultiworld
from Environments.PuckRealMulti.Sprites.GeneralSprite import GeneralSprite

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (135, 135, 0)

# Set the height and width of the screen

color = [BLUE, GREEN, RED, BLACK, BLUE, GREEN, RED, BLACK]


class PuckRealMultiFinalFrontend:
    # path update and path_lenght in seconds
    def __init__(self, env):
        # assert isinstance(env, tf_nn_sim.Environments.PuckRealMulti.Puckworld_Environment.PuckRealMultiworld)
        self.env = env
        self.drone_size = env.drone_radius
        # self.pos_to_pixel_x = env.area_pixel_x
        # self.pos_to_pixel_y = env.area_pixel_y
        self.debug_pixel_x = env.debug_pixel_x
        self.debug_pixel_y = env.debug_pixel_y
        self.m_to_pixel_x = env.area_pixel_x / env.total_meter_x
        self.m_to_pixel_y = env.area_pixel_y / env.total_meter_y
        self.action_width = int(env.debug_pixel_x / env.num_actions)
        self.action_height = self.action_width
        self.q_field_height = 11
        self.td_field_height = 11
        self.pause = False
        self.history_mode = False
        self.history_index = -1
        pygame.init()
        self.myfont = pygame.font.SysFont("monospace", 11)
        size = [env.area_pixel_x, env.area_pixel_y]
        self.screen = pygame.display.set_mode(size)
        self.all_sprites_list = pygame.sprite.Group()
        self.update_path = True
        self.max_skip = 200
        self.skip = 1
        self.call_count = 1

        action_file = "../../tf_nn_sim/Environments/PuckRealMulti/Images/action_NUMBER.png"
        self.sprites = []
        # h = self.action_height + self.q_field_height + self.td_field_height
        # for i in range(env.num_drones):
        #     self.sprites.append([])
        #     for j in range(env.num_actions):
        #         d = GeneralSprite(action_file.replace("NUMBER", str(j)), self.action_width, self.action_height,
        #                           env.area_pixel_x + (self.action_width * j), h * i)
        #         self.sprites[i].append(d)
        #         self.all_sprites_list.add(d)

        pygame.display.set_caption("Example code for the draw module")

    def key_update(self):
        next_frame = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    self.skip *= 2
                    self.skip = self.max_skip if self.skip > self.max_skip else self.skip
                    print self.skip
                elif event.key == pygame.K_DOWN:
                    self.skip /= 2
                    self.skip = 1 if self.skip < 1 else self.skip
                    print self.skip
                elif event.key == pygame.K_p:
                    self.pause = not self.pause
                    print "pause {0}".format(self.pause)
                elif event.key == pygame.K_h:
                    self.history_mode = not self.history_mode
                    print "history mode", self.history_mode
                    if not self.history_mode:
                        self.history_index = -1
                elif event.key == pygame.K_RIGHT:
                    next_frame = 1
                elif event.key == pygame.K_LEFT:
                    next_frame = -1
        return next_frame

    def get_key_action(self):
        action = 9
        pressed_key = pygame.key.get_pressed()
        if pressed_key[pygame.K_w]:
            action = 2
        elif pressed_key[pygame.K_s]:
            action = 3
        elif pressed_key[pygame.K_a]:
            action = 0
        elif pressed_key[pygame.K_d]:
            action = 1
        elif pressed_key[pygame.K_n]:
            action = 8
        return action

    def real_time_active(self):
        pressed_key = pygame.key.get_pressed()
        return pressed_key[pygame.K_r]

    def render(self, debug_data):
        while True:
            next_frame = self.key_update()
            if self.call_count % self.skip == 0:
                self.draw(debug_data, self.history_index)
                self.call_count = 0
            self.call_count += 1
            if self.history_mode:
                self.history_index += next_frame
                if self.history_index > -1:
                    self.history_index = -1
            if self.history_mode and (next_frame == 0):
                break
            if self.pause and next_frame != 0:
                break
            if not self.pause:
                break

    def draw_debug_data(self, screen, debug_data):
        for i in range(self.env.num_drones):
            if debug_data[i] is None:
                continue
            td = debug_data[i]["td_error"][0][0]
            # print "td data: ", td
            label = self.myfont.render("td_error: {:.4f}".format(td), 1, (0, 0, 0))
            screen.blit(label, (
                self.env.area_pixel_x, (self.action_height + self.q_field_height) * (i + 1) + self.td_field_height * i))
            velocity = debug_data[i]["velocity"]
            label = self.myfont.render("velocity: {:.4f}".format(velocity), 1, (0, 0, 0))
            screen.blit(label, (
                self.env.area_pixel_x + 200, (self.action_height + self.q_field_height) * (i + 1) + self.td_field_height * i))
            for j in range(self.env.num_actions):
                self.sprites[i][j].draw_background(screen, color[i], debug_data[i]["predict"][0] == j)
                q = debug_data[i]["q_values"][0]
                label = self.myfont.render("{:.4f}".format(q[j]), 1, (0, 0, 0))
                screen.blit(label, (self.sprites[i][j].rect.x, self.sprites[i][j].rect.y + self.action_height))

    def draw(self, debug_data, history_index=-1):
        # assert isinstance(self.env, tf_nn_sim.Environments.PuckRealMulti.Puckworld_Environment.PuckRealMultiworld)
        # self.ppx_history[drone_id].append(self.ppx[drone_id])
        # self.ppy_history[drone_id].append(self.ppy[drone_id])
        # self.pvx_history[drone_id].append(self.pvx[drone_id])
        # self.pvy_history[drone_id].append(self.pvy[drone_id])
        # self.tx_history[drone_id].append(self.tx[drone_id])
        # self.ty_history[drone_id].append(self.ty[drone_id])
        # self.orientation_history[drone_id].append(self.orientation[drone_id])
        self.screen.fill(WHITE)
        # self.draw_debug_data(self.screen, debug_data)

        # s = np.sqrt(self.env.pvx**2 + self.env.pvy**2)
        # # print (self.env.pvx / s), (self.env.pvy / s)
        # x_v, y_v = (self.env.pvx / (s+0.01)) * self.pos_to_pixel_x * 0.15, (self.env.pvy / (s + 0.01)) * 0.15 * self.pos_to_pixel_y
        # pygame.draw.circle(self.screen, color[0], [x, y], int(self.env.drone_radius * self.pos_to_pixel_x))
        # pygame.draw.line(self.screen, BLACK, [x, y], [x + x_v, y + y_v])
        for i in range(self.env.num_drones):
            if not self.env.training_goal:
                x, y = self.meter_to_pixel(self.env.tx_history[i][history_index], self.env.ty_history[i][history_index])
            else:
                x, y = self.meter_to_pixel(self.env.tx_train[i], self.env.ty_train[i])
            # x, y = self.pos_to_pixel(self.env.tx, self.env.ty)
            pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius * self.m_to_pixel_x))
            # print self.env.ppx[i], self.env.ppy[i]
            x, y = self.meter_to_pixel(self.env.ppx_history[i][history_index] - self.env.critical_radius,
                                       self.env.ppy_history[i][history_index] - self.env.critical_radius)
            s = pygame.Surface((self.env.critical_radius * self.m_to_pixel_x * 2,
                                self.env.critical_radius * self.m_to_pixel_x * 2))  # the size of your rect
            s.set_alpha(128)  # alpha level
            s.fill((255, 255, 255))
            pygame.draw.circle(s, color[i],
                               [int(self.env.critical_radius * self.m_to_pixel_x),
                                int(self.env.critical_radius * self.m_to_pixel_y)],
                               int(self.env.critical_radius * self.m_to_pixel_x))
            self.screen.blit(s, [x, y])
            x, y = self.meter_to_pixel(self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index])
            pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius * 1.0 * self.m_to_pixel_x))
            self.draw_line(BLACK, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                           self.wrap_angle(self.env.orientation_history[i][history_index]), self.env.drone_radius * 1.5)
            self.draw_line(BLACK, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                           self.wrap_angle(self.env.orientation_history[i][history_index] + np.pi), self.env.drone_radius)
            self.draw_line(BLACK, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                           self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.get_velocity_angle(i)),
                           self.env.drone_radius * 2.0)
            self.draw_line(RED, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                           self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.get_goal_angle(i)), self.env.get_goal_distance(i))
            if self.env.get_obstacle_angle(i) is not None:
                self.draw_line(BLUE, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                               self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.get_obstacle_angle(i)),
                               self.env.get_obstacle_distance(i))
            # self.draw_line(BLUE, self.env.ppx[i], self.env.ppy[i],
            #                self.wrap_angle(self.env.orientation[i] + self.env.goal_angle[i]), self.env.goal_distance[i])


            if len(self.env.predict_obstacle_angle) > i:
                try:
                    self.draw_line(RED, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                                   self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.predict_obstacle_angle[i]),
                                   self.env.drone_radius * 4.0)
                    self.draw_end_circle(RED, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                                         self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.predict_obstacle_angle[i]),
                                         self.env.drone_radius * 4.0, 0.025)
                except TypeError:
                    print "typeError :("

            if i in self.env.predict_obstacle_orientation:
                c_id = self.env.closes_drone(i)
                if c_id is not None:
                    self.draw_line(BLACK, self.env.ppx_history[c_id][history_index], self.env.ppy_history[c_id][history_index],
                                   self.wrap_angle(
                                       self.env.orientation_history[i][history_index] + self.env.predict_obstacle_orientation[i]),
                                   self.env.drone_radius * 4.0)
                    self.draw_end_circle(BLACK, self.env.ppx_history[c_id][history_index],
                                         self.env.ppy_history[c_id][history_index],
                                         self.wrap_angle(self.env.orientation_history[i][history_index] +
                                                         self.env.predict_obstacle_orientation[i]),
                                         self.env.drone_radius * 4.0, 0.025)

            for test in range(len(self.env.particles_goal[i])):
                x, y = self.meter_to_pixel(self.env.particles_goal[i][test][0], self.env.particles_goal[i][test][1])
                pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius/10.0 * self.m_to_pixel_x))

            for test in range(len(self.env.particles_obs[i])):
                x, y = self.meter_to_pixel(self.env.particles_obs[i][test][0], self.env.particles_obs[i][test][1])
                yaw = self.env.particles_obs[i][test][2]
                self.draw_line(BLACK, self.env.particles_obs[i][test][0], self.env.particles_obs[i][test][1], yaw, self.env.drone_radius * 1.5)
                # x, y = self.pos_to_pixel(self.env.tx, self.env.ty)
                pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius/3.0 * self.m_to_pixel_x))

            if i in self.env.particle_obs_predict:
                # print self.env.particle_obs_predict[i][0], self.env.particle_obs_predict[i][1]
                x, y = self.meter_to_pixel(self.env.particle_obs_predict[i][0], self.env.particle_obs_predict[i][1])
                pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius * 1.0 * self.m_to_pixel_x))

            if i in self.env.particle_goal_predict:
                x, y = self.meter_to_pixel(self.env.particle_goal_predict[i][0], self.env.particle_goal_predict[i][1])
                pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius * 1.0 * self.m_to_pixel_x))
                pygame.draw.circle(self.screen, RED, [x, y], int(self.env.drone_radius * 0.4 * self.m_to_pixel_x))

            if len(self.env.predict_goal_angle) > i:
                try:
                    self.draw_line(BLUE, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                                   self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.predict_goal_angle[i]),
                                   self.env.drone_radius * 4.0)
                    self.draw_end_circle(BLUE, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                                         self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.predict_goal_angle[i]),
                                         self.env.drone_radius * 4.0, 0.025)
                except TypeError:
                    print "typeError :("

            if len(self.env.predict_goal_angle_noise) > i:
                try:
                    self.draw_line(GREEN, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                                   self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.predict_goal_angle_noise[i]),
                                   self.env.drone_radius * 4.0)
                    self.draw_end_circle(GREEN, self.env.ppx_history[i][history_index], self.env.ppy_history[i][history_index],
                                         self.wrap_angle(self.env.orientation_history[i][history_index] + self.env.predict_goal_angle_noise[i]),
                                         self.env.drone_radius * 4.0, 0.025)
                except TypeError:
                    print "typeError :("

        # self.screen.set_alpha(100)
        p1_x, p1_y = self.meter_to_pixel(self.env.x_start, self.env.y_start)
        p2_x, p2_y = self.meter_to_pixel(self.env.x_end, self.env.y_start)
        p3_x, p3_y = self.meter_to_pixel(self.env.x_end, self.env.y_end)
        p4_x, p4_y = self.meter_to_pixel(self.env.x_start, self.env.y_end)

        pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p2_x, p2_y])
        pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p2_x, p2_y])
        pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p4_x, p4_y])
        pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p4_x, p4_y])
        self.all_sprites_list.draw(self.screen)
        pygame.display.flip()

    def draw_end_circle(self, color, pos_x, pos_y, angle, length, radius):
        x, y = self.meter_to_pixel(pos_x, pos_y)
        x_o = np.math.cos(angle)
        y_o = np.math.sin(angle)
        x_v, y_v = int(x_o * self.m_to_pixel_x * length), int(y_o * length * self.m_to_pixel_x)
        pygame.draw.circle(self.screen, color, [x + x_v, y + y_v], int(radius * self.m_to_pixel_x))

    def draw_line(self, color, pos_x, pos_y, angle, length):
        x, y = self.meter_to_pixel(pos_x, pos_y)
        x_o = np.math.cos(angle)
        y_o = np.math.sin(angle)
        x_v, y_v = x_o * self.m_to_pixel_x * length, y_o * length * self.m_to_pixel_x
        pygame.draw.line(self.screen, color, [x, y], [x + x_v, y + y_v])

    def wrap_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def meter_to_pixel(self, x, y):
        x = int((x + self.env.total_meter_x / 2.0) * self.m_to_pixel_x)
        y = int((y + self.env.total_meter_y / 2.0) * self.m_to_pixel_y)

        return x, y

    # def meter_to_pixel(self, x, y):
    #     x = int(x * self.pos_to_pixel_x)
    #     y = int(y * self.pos_to_pixel_y)
    #
    #     return x, y

    def close_window(self):
        # Be IDLE friendly
        pygame.quit()
