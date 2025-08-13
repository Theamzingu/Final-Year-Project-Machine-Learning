import pygame
from entity import Entity
from constants import *
from sprites import FruitSprites

class Fruit(Entity):
    def __init__(self, node, lvl = 0):
        Entity.__init__(self, node)
        self.name = FRUIT
        self.color = GREEN
        self.lifespan = 10
        self.timer = 0
        self.destroy = False
        self.points = 100 + lvl * 20
        self.setBetweenNodes(RIGHT)
        self.sprites = FruitSprites(self, lvl)

    def update(self, dt):
        self.timer += dt
        if self.timer >= self.lifespan:
            self.destroy = True