# Define the colors we will use in RGB format
import pygame
import sys

import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (135, 135, 0)

# Set the height and width of the screen

color = [BLUE, GREEN, RED, BLACK, BLUE, GREEN, RED, BLACK]


class PuckFrontend:
    # path update and path_lenght in seconds
    def __init__(self, env):
        assert isinstance(env, src.Environments.Puck.Puckworld_Environment.Puckworld)
        self.env = env
        self.drone_size = env.drone_radius
        self.meter_x = env.meter_x
        self.meter_y = env.meter_y
        self.x_offset = (env.meter_x - env.wall_size_x) / 2.0
        self.y_offset = (env.meter_y - env.wall_size_y) / 2.0
        self.m_to_pixel_x = env.pixel_x / env.meter_x
        self.m_to_pixel_y = env.pixel_y / env.meter_y
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
        assert isinstance(self.env, src.Environments.Puck.Puckworld_Environment.Puckworld)
        self.screen.fill(WHITE)

        x, y = self.meter_to_pixel(self.env.ppx, self.env.ppy)
        s = np.sqrt(self.env.pvx**2 + self.env.pvy**2)
        # print (self.env.pvx / s), (self.env.pvy / s)
        x_v, y_v = (self.env.pvx / (s+0.01)) * self.m_to_pixel_x * 0.15, (self.env.pvy / (s+0.01)) * 0.15 * self.m_to_pixel_y
        pygame.draw.circle(self.screen, color[0], [x, y], int(self.env.drone_radius * self.m_to_pixel_x))
        pygame.draw.line(self.screen, BLACK, [x, y], [x + x_v, y + y_v])
        x, y = self.meter_to_pixel(self.env.tx, self.env.ty)
        pygame.draw.circle(self.screen, color[1], [x, y], int(self.env.drone_radius * self.m_to_pixel_x))

        x, y = self.meter_to_pixel(self.env.tx2 - self.env.bad_radius, self.env.ty2 - self.env.bad_radius)
        s = pygame.Surface((self.env.bad_radius * self.m_to_pixel_x * 2,
                            self.env.bad_radius * self.m_to_pixel_y * 2))  # the size of your rect
        s.set_alpha(128)  # alpha level
        s.fill((255, 255, 255))
        pygame.draw.circle(s, color[2],
                           [int(self.env.bad_radius * self.m_to_pixel_x), int(self.env.bad_radius * self.m_to_pixel_y)],
                           int(self.env.bad_radius * self.m_to_pixel_y))
        self.screen.blit(s, [x, y])
        x, y = self.meter_to_pixel(self.env.tx2, self.env.ty2)
        pygame.draw.circle(self.screen, color[2], [x, y], int(self.env.drone_radius/2 * self.m_to_pixel_y))
        # self.screen.set_alpha(100)
        p1_x, p1_y = self.meter_to_pixel(self.env.x_start, self.env.y_start)
        p2_x, p2_y = self.meter_to_pixel(self.env.x_end, self.env.y_start)
        p3_x, p3_y = self.meter_to_pixel(self.env.x_end, self.env.y_end)
        p4_x, p4_y = self.meter_to_pixel(self.env.x_start, self.env.y_end)

        pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p2_x, p2_y])
        pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p2_x, p2_y])
        pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p4_x, p4_y])
        pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p4_x, p4_y])

        pygame.display.flip()

    def meter_to_pixel(self, x, y):
        x = int((x + self.meter_x/2.0) * self.m_to_pixel_x)
        y = int((y + self.meter_y/2.0) * self.m_to_pixel_y)

        return x, y

    def close_window(self):
        # Be IDLE friendly
        pygame.quit()
