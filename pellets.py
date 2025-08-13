import pygame
from vectors import Vector
from constants import *
import numpy as np
from copy import deepcopy

class Pellet:
    def __init__(self, row, col):
        self.name = PELLET
        self.pos = Vector((col + 1.5) * TILE_WIDTH, row * TILE_HEIGHT)
        self.color = WHITE
        self.rad = int(2 * TILE_WIDTH / 16)
        self.collideRad = int(2 * TILE_WIDTH / 16)
        self.points = 10
        self.visible = True

    def render(self, screen):
        if self.visible:
            adjust = Vector(TILE_WIDTH, TILE_HEIGHT) / 2
            p = self.pos + adjust
            pygame.draw.circle(screen, self.color, p.asInt(), self.rad)

class PowerPellet(Pellet):
    def __init__(self, row, col):
        Pellet.__init__(self, row, col)
        self.name = POWER_PELLET
        self.rad = int(8 * TILE_WIDTH / 16)
        self.points = 50
        self.flashTime = 0.2
        self.timer = 0
    
    def update(self, dt):
        self.timer += dt
        if self.timer >= self.flashTime:
            self.visible = not self.visible
            self.timer = 0

class PelletGroup:
    def __init__(self, pelletfile):
        self.pelletList = []
        self.powerpellets = []
        self.createPelletList(pelletfile)
        self.numEaten = 0

    def __deepcopy__(self, memo):
        new_obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_obj
        setattr(new_obj, 'pelletList', self.pelletList.copy())
        setattr(new_obj, 'powerpellets', self.powerpellets.copy())
        setattr(new_obj, 'numEaten', self.numEaten)
        return new_obj
    
    def update(self, dt):
        for powerpellet in self.powerpellets:
            powerpellet.update(dt)

    def createPelletList(self, pelletfile):
        data = self.readPelletfile(pelletfile)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in ['.', '+']:
                    self.pelletList.append(Pellet(row, col))
                elif data[row][col] in ['P', 'p']:
                    pp = PowerPellet(row, col)
                    self.pelletList.append(pp)
                    self.powerpellets.append(pp)

    def readPelletfile(self, textfile):
        return np.loadtxt(textfile, dtype = '<U1')
    
    def isEmpty(self):
        if len(self.pelletList) == 0:
            return True
        return False

    def render(self, screen):
        for pellet in self.pelletList:
            pellet.render(screen)