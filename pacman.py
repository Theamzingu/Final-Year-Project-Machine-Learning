import pygame
from pygame.locals import *
from vectors import Vector
from constants import *
from entity import Entity
from sprites import PacmanSprites
import heapq

class Pacman(Entity):
    def __init__(self, node):
        Entity.__init__(self, node)
        self.name = PACMAN
        self.color = YELLOW
        self.dir = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.sprites = PacmanSprites(self)

    def update(self, dt, key_pressed):
        self.sprites.update(dt)
        self.pos += self.dirs[self.dir] * self.speed * dt
        dir = self.getValidKey(key_pressed)
        if self.overshotTarget():
            self.node = self.target
            if self.node.neighbors[PORTAL] is not None:
                self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(dir)
            if self.target is not self.node:
                self.dir = dir
            else:
                self.target = self.getNewTarget(self.dir)
            if self.target is self.node:
                self.dir = STOP
            self.setPos()
        else:
            if self.oppositeDir(dir):
                self.reverseDir()

    def getValidKey(self, key_pressed):
        if key_pressed[K_UP]:
            return UP
        if key_pressed[K_DOWN]:
            return DOWN
        if key_pressed[K_LEFT]:
            return LEFT
        if key_pressed[K_RIGHT]:
            return RIGHT
        return STOP
    
    def eatPellets(self, pelletList):
        for pellet in pelletList:
            if self.collideCheck(pellet):
                return pellet
        return None

    def collideGhost(self, ghost):
        return self.collideCheck(ghost)
    
    def collideCheck(self, other):
        d = self.pos - other.pos
        dSquared = d.magnitudeSquared()
        rSquared = (self.collideRad + other.collideRad) ** 2
        if dSquared <= rSquared:
            return True
        return False

    def reset(self):
        Entity.reset(self)
        self.dir = LEFT
        self.setBetweenNodes(LEFT)
        self.alive = True
        self.image = self.sprites.getStartImage()
        self.sprites.reset()

    def dead(self):
        self.alive = False
        self.dir = STOP