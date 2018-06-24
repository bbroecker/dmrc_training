import collections
import copy
import pygame

import math


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (135, 135, 0)

class GeneralSprite(pygame.sprite.Sprite):
    # This class represents a car. It derives from the "Sprite" class in Pygame.

    def __init__(self, file_path, width, height, pos_x, pos_y):
        # Call the parent class (Sprite) constructor
        pygame.sprite.Sprite.__init__(self)
        tmp = pygame.image.load(file_path).convert_alpha()

        self.image = pygame.Surface([width, height]).convert_alpha()

        pygame.transform.smoothscale(tmp.convert_alpha(), (width, height), self.image)
        self.image = self.image.convert_alpha()
        self.master_img = self.image.copy()

        # Fetch the rectangle object that has the dimensions of the image.
        self.rect = self.image.get_rect()
        self.rect.width = width
        self.rect.height = height
        self.rect.x = pos_x
        self.rect.y = pos_y
        self.count = 0
        self.s = pygame.Surface((width,height))  # the size of your rect
        self.s.set_alpha(128)  # alpha level
        self.s.fill((255, 255, 255))

    def draw_background(self, screen, color, select):
        pygame.draw.rect(self.s, color if select else WHITE, (0, 0, self.rect.width, self.rect.height))
        screen.blit(self.s, [self.rect.x, self.rect.y])

    def rotate(self, theta):
        angle = -math.degrees(theta)
        # self.rect.centerx = -(self.y * METRES_TO_PIXELS)
        # self.rect.centery = SCREENHEIGHT-(self.x * METRES_TO_PIXELS)
        old_center = self.rect.center
        self.image = pygame.transform.rotate(self.master_img, angle)
        self.rect = self.image.get_rect()
        self.rect.center = old_center
