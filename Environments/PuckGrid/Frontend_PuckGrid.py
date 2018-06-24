# Define the colors we will use in RGB format
import pygame
import sys

import tf_nn_sim.Environments.PuckGrid.PuckGrid_Environment

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (135, 135, 0)

# Set the height and width of the screen

color = [BLUE, GREEN, RED, BLACK, BLUE, GREEN, RED, BLACK]


class PuckGridFrontend:
    # path update and path_lenght in seconds
    def __init__(self, env):
        assert isinstance(env, tf_nn_sim.Environments.PuckGrid.PuckGrid_Environment.PuckGrid)
        self.env = env
        self.index_to_pixel_x = env.pixel_x / env.grid_size_x
        self.index_to_pixel_y = env.pixel_y / env.grid_size_y
        pygame.init()
        size = [env.pixel_x, env.pixel_y]
        self.screen = pygame.display.set_mode(size)
        self.all_sprites_list = pygame.sprite.Group()
        self.max_skip = 200
        self.skip = 1
        self.call_count = 1
        self.key_update_rate = True

        # self.cf_sprites = []
        # for i in range(len(self.simulator.m_drones)):
        #     d = DroneSprite(color[i], int(self.drone_size * self.m_to_pixel), int(self.drone_size * self.m_to_pixel), int(path_length/ path_update))
        #     self.cf_sprites.append(d)
        #     self.all_sprites_list.add(d)

        pygame.display.set_caption("Example code for the draw module")

    def key_update(self):
        if self.key_update_rate:
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

    def get_board_action(self):
        action = -1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
                elif event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_q:
                    action = -10
        return action

    def render(self):
        self.key_update()
        if self.call_count % self.skip == 0:
            self.draw()
            self.call_count = 0
        self.call_count += 1

    def draw(self):
        assert isinstance(self.env, tf_nn_sim.Environments.PuckGrid.PuckGrid_Environment.PuckGrid)
        self.screen.fill(WHITE)
        for i in range(0, self.env.x_end + 1):
            pygame.draw.line(self.screen, BLACK, [self.index_to_pixel_x * i, self.env.y_start * self.index_to_pixel_y],
                             [self.index_to_pixel_x * i, (self.env.y_end + 1) * self.index_to_pixel_y])

        for i in range(0, self.env.y_end + 1):
            pygame.draw.line(self.screen, BLACK, [self.env.x_start * self.index_to_pixel_x, self.index_to_pixel_y * i],
                             [(self.env.x_end + 1) * self.index_to_pixel_x, self.index_to_pixel_y * i])

        x, y = self.env.ppx * self.index_to_pixel_x, self.env.ppy * self.index_to_pixel_y
        pygame.draw.rect(self.screen, color[0], [x, y, self.index_to_pixel_x, self.index_to_pixel_y])

        x, y = self.env.ox * self.index_to_pixel_x, self.env.oy * self.index_to_pixel_y
        pygame.draw.rect(self.screen, color[2], [x, y, self.index_to_pixel_x, self.index_to_pixel_y])

        x, y = self.env.tx * self.index_to_pixel_x, self.env.ty * self.index_to_pixel_y
        pygame.draw.rect(self.screen, color[1], [x, y, self.index_to_pixel_x, self.index_to_pixel_y])

        # s = np.sqrt(self.env.pvx ** 2 + self.env.pvy ** 2)
        # # print (self.env.pvx / s), (self.env.pvy / s)
        # x_v, y_v = (self.env.pvx / (s + 0.01)) * self.m_to_pixel_x * 0.15, (
        # self.env.pvy / (s + 0.01)) * 0.15 * self.m_to_pixel_y
        #
        # pygame.draw.line(self.screen, BLACK, [x, y], [x + x_v, y + y_v])
        # x, y = self.meter_to_pixel(self.env.tx, self.env.ty)
        # pygame.draw.circle(self.screen, color[1], [x, y], int(self.env.drone_radius * self.m_to_pixel_x))
        #
        # x, y = self.meter_to_pixel(self.env.tx2 - self.env.bad_radius, self.env.ty2 - self.env.bad_radius)
        # s = pygame.Surface((self.env.bad_radius * self.m_to_pixel_x * 2,
        #                     self.env.bad_radius * self.m_to_pixel_y * 2))  # the size of your rect
        # s.set_alpha(128)  # alpha level
        # s.fill((255, 255, 255))
        # pygame.draw.circle(s, color[2],
        #                    [int(self.env.bad_radius * self.m_to_pixel_x), int(self.env.bad_radius * self.m_to_pixel_y)],
        #                    int(self.env.bad_radius * self.m_to_pixel_y))
        # self.screen.blit(s, [x, y])
        # x, y = self.meter_to_pixel(self.env.tx2, self.env.ty2)
        # pygame.draw.circle(self.screen, color[2], [x, y], int(self.env.drone_radius / 2 * self.m_to_pixel_y))
        # # self.screen.set_alpha(100)
        # p1_x, p1_y = self.meter_to_pixel(self.env.x_start, self.env.y_start)
        # p2_x, p2_y = self.meter_to_pixel(self.env.x_end, self.env.y_start)
        # p3_x, p3_y = self.meter_to_pixel(self.env.x_end, self.env.y_end)
        # p4_x, p4_y = self.meter_to_pixel(self.env.x_start, self.env.y_end)
        #
        # pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p2_x, p2_y])
        # pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p2_x, p2_y])
        # pygame.draw.line(self.screen, BLACK, [p3_x, p3_y], [p4_x, p4_y])
        # pygame.draw.line(self.screen, BLACK, [p1_x, p1_y], [p4_x, p4_y])

        pygame.display.flip()


    def close_window(self):
        # Be IDLE friendly
        pygame.quit()
