#
# Copyright (c) 2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import pygame
from pygame.locals import HWSURFACE, DOUBLEBUF


class Renderer(object):
    def __init__(self):
        self.size = (1, 1)
        self.screen = None
        self.clock = pygame.time.Clock()
        self.display = pygame.display
        self.fps = 30
        self.pressed_keys = []
        self.is_open = False

    def create_screen(self, width, height):
        """
        Creates a pygame window
        :param width: the width of the window
        :param height: the height of the window
        :return: None
        """
        self.size = (width, height)
        self.screen = self.display.set_mode(self.size, HWSURFACE | DOUBLEBUF)
        self.display.set_caption("Coach")
        self.is_open = True

    def normalize_image(self, image):
        """
        Normalize image values to be between 0 and 255
        :param image: 2D/3D array containing an image with arbitrary values
        :return: the input image with values rescaled to 0-255
        """
        image_min, image_max = image.min(), image.max()
        return 255.0 * (image - image_min) / (image_max - image_min)

    def render_image(self, image):
        """
        Render the given image to the pygame window
        :param image: a grayscale or color image in an arbitrary size. assumes that the channels are the last axis
        :return: None
        """
        if self.is_open:
            if len(image.shape) == 2:
                image = np.stack([image] * 3)
            if len(image.shape) == 3:
                if image.shape[0] == 3 or image.shape[0] == 1:
                    image = np.transpose(image, (1, 2, 0))
            surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            surface = pygame.transform.scale(surface, self.size)
            self.screen.blit(surface, (0, 0))
            self.display.flip()
            self.clock.tick()
            self.get_events()

    def get_events(self):
        """
        Get all the window events in the last tick and reponse accordingly
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.pressed_keys.append(event.key)
                # esc pressed
                if event.key == pygame.K_ESCAPE:
                    self.close()
            elif event.type == pygame.KEYUP:
                if event.key in self.pressed_keys:
                    self.pressed_keys.remove(event.key)
            elif event.type == pygame.QUIT:
                self.close()

    def get_key_names(self, key_ids):
        """
        Get the key name for each key index in the list
        :param key_ids: a list of key id's
        :return: a list of key names corresponding to the key id's
        """
        return [pygame.key.name(key_id) for key_id in key_ids]

    def close(self):
        """
        Close the pygame window
        :return: None
        """
        self.is_open = False
        pygame.quit()
