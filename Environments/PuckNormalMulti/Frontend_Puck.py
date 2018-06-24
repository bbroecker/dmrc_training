# Define the colors we will use in RGB format
import pygame
import sys

import numpy as np
from Environments.PuckNormalMulti.Sprites.GeneralSprite import GeneralSprite

import tf_nn_sim.Environments.PuckNormalMulti.Puckworld_Environment

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (135, 135, 0)

# Set the height and width of the screen

color = [BLUE, GREEN, RED, BLACK, BLUE, GREEN, RED, BLACK]


class PuckNormalMultiFrontend:
    # path update and path_lenght in seconds
    def __init__(self, env):
        assert isinstance(env, tf_nn_sim.Environments.PuckNormalMulti.Puckworld_Environment.PuckNormalMultiworld)
        self.env = env
        self.drone_size = env.drone_radius
        self.pos_to_pixel_x = env.area_pixel_x
        self.pos_to_pixel_y = env.area_pixel_y
        self.debug_pixel_x = env.debug_pixel_x
        self.debug_pixel_y = env.debug_pixel_y
        self.action_width = int(env.debug_pixel_x / env.num_actions)
        self.action_height = self.action_width
        self.q_field_height = 15
        self.td_field_height = 15
        self.pause = False
        pygame.init()
        self.myfont = pygame.font.SysFont("monospace", 15)
        size = [env.area_pixel_x + self.debug_pixel_x, env.area_pixel_y]
        self.screen = pygame.display.set_mode(size)
        self.all_sprites_list = pygame.sprite.Group()
        self.update_path = True
        self.max_skip = 200
        self.skip = 1
        self.call_count = 1

        action_file = "../Environments/PuckNormalMulti/Images/action_NUMBER.png"
        self.sprites = []
        h = self.action_height + self.q_field_height + self.td_field_height
        for i in range(env.num_drones):
            self.sprites.append([])
            for j in range(env.num_actions):
                d = GeneralSprite(action_file.replace("NUMBER", str(j)), self.action_width, self.action_height,
                                  env.area_pixel_x + (self.action_width * j), h * i)
                self.sprites[i].append(d)
                self.all_sprites_list.add(d)

        pygame.display.set_caption("Example code for the draw module")

    def key_update(self):
        next_frame = False
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
                elif event.key == pygame.K_RIGHT:
                    next_frame = True
        return next_frame

    def render(self, debug_data):
        while True:
            next_frame = self.key_update()
            if self.call_count % self.skip == 0:
                self.draw(debug_data)
                self.call_count = 0
            self.call_count += 1
            if not self.pause or next_frame:
                break

    def draw_debug_data(self, screen, debug_data):
        for i in range(self.env.num_drones):
            if debug_data[i] is None:
                continue
            td = debug_data[i]["td_error"][0]
            label = self.myfont.render("td_error: {:.4f}".format(td), 1, (0, 0, 0))
            screen.blit(label, (
            self.pos_to_pixel_x, (self.action_height + self.q_field_height) * (i + 1) + self.td_field_height * i))
            for j in range(self.env.num_actions):
                self.sprites[i][j].draw_background(screen, color[i], debug_data[i]["predict"][0] == j)
                q = debug_data[i]["q_values"][0]
                label = self.myfont.render("{:.4f}".format(q[j]), 1, (0, 0, 0))
                screen.blit(label, (self.sprites[i][j].rect.x, self.sprites[i][j].rect.y + self.action_height))

    def get_key_action(self):
        action = 4
        pressed_key = pygame.key.get_pressed()
        if pressed_key[pygame.K_w]:
            action = 2
        elif pressed_key[pygame.K_s]:
            action = 3
        elif pressed_key[pygame.K_a]:
            action = 0
        elif pressed_key[pygame.K_d]:
            action = 1
        return action

    def real_time_active(self):
        pressed_key = pygame.key.get_pressed()
        return pressed_key[pygame.K_r]

    def draw(self, debug_data):
        assert isinstance(self.env, tf_nn_sim.Environments.PuckNormalMulti.Puckworld_Environment.PuckNormalMultiworld)
        self.screen.fill(WHITE)
        self.draw_debug_data(self.screen, debug_data)

        # s = np.sqrt(self.env.pvx**2 + self.env.pvy**2)
        # # print (self.env.pvx / s), (self.env.pvy / s)
        # x_v, y_v = (self.env.pvx / (s+0.01)) * self.pos_to_pixel_x * 0.15, (self.env.pvy / (s + 0.01)) * 0.15 * self.pos_to_pixel_y
        # pygame.draw.circle(self.screen, color[0], [x, y], int(self.env.drone_radius * self.pos_to_pixel_x))
        # pygame.draw.line(self.screen, BLACK, [x, y], [x + x_v, y + y_v])
        for i in range(self.env.num_drones):
            x, y = self.pos_to_pixel(self.env.tx[i], self.env.ty[i])
            # x, y = self.pos_to_pixel(self.env.tx, self.env.ty)
            pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius * self.pos_to_pixel_x))

            x, y = self.pos_to_pixel(self.env.ppx[i] - self.env.bad_radius, self.env.ppy[i] - self.env.bad_radius)
            s = pygame.Surface((self.env.bad_radius * self.pos_to_pixel_x * 2,
                                self.env.bad_radius * self.pos_to_pixel_y * 2))  # the size of your rect
            s.set_alpha(128)  # alpha level
            s.fill((255, 255, 255))
            pygame.draw.circle(s, color[i],
                               [int(self.env.bad_radius * self.pos_to_pixel_x),
                                int(self.env.bad_radius * self.pos_to_pixel_y)],
                               int(self.env.bad_radius * self.pos_to_pixel_y))
            self.screen.blit(s, [x, y])
            x, y = self.pos_to_pixel(self.env.ppx[i], self.env.ppy[i])
            pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius / 2 * self.pos_to_pixel_y))

            if len(self.env.angle_obstacle) > i:
                try:
                    x_o = np.math.cos(self.env.angle_obstacle[i])
                    y_o = np.math.sin(self.env.angle_obstacle[i])
                    x_v, y_v = x_o * self.pos_to_pixel_x * 0.15, y_o * 0.15 * self.pos_to_pixel_y
                    pygame.draw.line(self.screen, RED, [x, y], [x + x_v, y + y_v])
                except TypeError:
                    print "typeError :("

        # self.screen.set_alpha(100)
        p1_x, p1_y = self.pos_to_pixel(self.env.x_start, self.env.y_start)
        p2_x, p2_y = self.pos_to_pixel(self.env.x_end, self.env.y_start)
        p3_x, p3_y = self.pos_to_pixel(self.env.x_end, self.env.y_end)
        p4_x, p4_y = self.pos_to_pixel(self.env.x_start, self.env.y_end)

        pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p2_x, p2_y])
        pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p2_x, p2_y])
        pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p4_x, p4_y])
        pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p4_x, p4_y])
        self.all_sprites_list.draw(self.screen)
        pygame.display.flip()

    def pos_to_pixel(self, x, y):
        x = int(x * self.pos_to_pixel_x)
        y = int(y * self.pos_to_pixel_y)

        return x, y

    def close_window(self):
        # Be IDLE friendly
        pygame.quit()
