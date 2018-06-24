# Define the colors we will use in RGB format
import pygame
import sys

import numpy as np

import tf_nn_sim.Environments.PuckNormalRandom.Puckworld_Environment

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (135, 135, 0)

# Set the height and width of the screen

color = [BLUE, GREEN, RED, BLACK, BLUE, GREEN, RED, BLACK]


class PuckNormalRandomFrontend:
    # path update and path_lenght in seconds
    def __init__(self, env):
        assert isinstance(env, tf_nn_sim.Environments.PuckNormalRandom.Puckworld_Environment.PuckNormalRandomworld)
        self.env = env
        self.drone_size = env.drone_radius
        self.pos_to_pixel_x = env.pixel_x
        self.pos_to_pixel_y = env.pixel_y
        pygame.init()
        size = [env.pixel_x, env.pixel_y]
        self.screen = pygame.display.set_mode(size)
        self.all_sprites_list = pygame.sprite.Group()
        self.update_path = True
        self.max_skip = 200
        self.skip = 1
        self.call_count = 1

        # self.cf_sprites = []
        # for i in range(len(self.simulator.m_drones)):
        #     d = DroneSprite(color[i], int(self.drone_size * self.m_to_pixel), int(self.drone_size * self.m_to_pixel), int(path_length/ path_update))
        #     self.cf_sprites.append(d)
        #     self.all_sprites_list.add(d)

        pygame.display.set_caption("Example code for the draw module")

    def key_update(self):
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


    def render(self):
        self.key_update()
        if self.call_count % self.skip == 0:
            self.draw()
            self.call_count = 0
        self.call_count += 1


    def draw(self):
        assert isinstance(self.env, tf_nn_sim.Environments.PuckNormalRandom.Puckworld_Environment.PuckNormalRandomworld)
        self.screen.fill(WHITE)



        # s = np.sqrt(self.env.pvx**2 + self.env.pvy**2)
        # # print (self.env.pvx / s), (self.env.pvy / s)
        # x_v, y_v = (self.env.pvx / (s+0.01)) * self.pos_to_pixel_x * 0.15, (self.env.pvy / (s + 0.01)) * 0.15 * self.pos_to_pixel_y
        # pygame.draw.circle(self.screen, color[0], [x, y], int(self.env.drone_radius * self.pos_to_pixel_x))
        # pygame.draw.line(self.screen, BLACK, [x, y], [x + x_v, y + y_v])
        for i in range(self.env.num_drones):
            # x, y = self.pos_to_pixel(self.env.tx[i], self.env.ty[i])
            # # x, y = self.pos_to_pixel(self.env.tx, self.env.ty)
            # pygame.draw.circle(self.screen, color[i], [x, y], int(self.env.drone_radius * self.pos_to_pixel_x))

            # x, y = self.pos_to_pixel(self.env.ppx[i] - self.env.bad_radius, self.env.ppy[i] - self.env.bad_radius)
            # s = pygame.Surface((self.env.bad_radius * self.pos_to_pixel_x * 2,
            #                     self.env.bad_radius * self.pos_to_pixel_y * 2))  # the size of your rect
            # s.set_alpha(128)  # alpha level
            # s.fill((255, 255, 255))
            # pygame.draw.circle(s, color[i],
            #                    [int(self.env.bad_radius * self.pos_to_pixel_x), int(self.env.bad_radius * self.pos_to_pixel_y)],
            #                    int(self.env.bad_radius * self.pos_to_pixel_y))
            # self.screen.blit(s, [x, y])
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

        pygame.display.flip()

    def pos_to_pixel(self, x, y):
        x = int(x * self.pos_to_pixel_x)
        y = int(y * self.pos_to_pixel_y)

        return x, y

    def close_window(self):
        # Be IDLE friendly
        pygame.quit()
