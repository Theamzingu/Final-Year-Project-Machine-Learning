import pygame
from pygame.locals import *
from vectors import Vector
from constants import *
import random

class Entity:
    def __init__(self, node):
        self.name = None
        self.dirs = {UP:Vector(0, -1), DOWN:Vector(0, 1), LEFT:Vector(-1, 0), RIGHT:Vector(1, 0), STOP:Vector()}
        self.dir = STOP
        self.setSpeed(100)
        self.rad = 10
        self.collideRad = 5
        self.color = WHITE
        self.visible = True
        self.disablePortal = False
        self.goal = None
        self.dirMethod = self.randomDir
        self.setStartNode(node)
        self.rng = random.Random()

    def setStartNode(self, node):
        self.node = node
        self.startNode = node
        self.target = node
        self.setPos()

    def setPos(self):
        self.pos = self.node.pos.copy()
    
    def validDir(self, dir):
        if dir != STOP:
            if self.name in self.node.access[dir]:
                if self.node.neighbors[dir] is not None:
                    return True
        return False

    def getNewTarget(self, dir):
        if self.validDir(dir):
            return self.node.neighbors[dir]
        return self.node

    def overshotTarget(self):
        if self.target is not None:
            vec1 = self.target.pos - self.node.pos
            vec2 = self.pos - self.node.pos
            nodeToTarget = vec1.magnitudeSquared()
            nodeToSelf = vec2.magnitudeSquared()
            return nodeToSelf >= nodeToTarget
        return False

    def reverseDir(self):
        self.dir *= -1
        temp = self.node
        self.node = self.target
        self.target = temp
        
    def oppositeDir(self, dir):
        if dir is not STOP:
            if dir == self.dir * -1:
                return True
        return False

    def setSpeed(self, speed):
        self.speed = speed * TILE_WIDTH / 16

    def render(self, screen):
        if self.visible:
            if self.sprites is not None:
                adjust = Vector(TILE_WIDTH, TILE_HEIGHT) / 2
                p = self.pos - adjust
                screen.blit(self.sprites.image, p.asTuple())
            else:
                p = self.pos.asInt()
                pygame.draw.circle(screen, self.color, p, self.rad)
        
    def update(self, dt):
        self.pos += self.dirs[self.dir] * self.speed * dt
        if self.overshotTarget():
            self.node = self.target
            dirs = self.validDirs()
            dir = self.dirMethod(dirs)
            if not self.disablePortal:
                if self.node.neighbors[PORTAL] is not None:
                    self.node = self.node.neighbors[PORTAL]
            self.target = self.getNewTarget(dir)
            if self.target is not self.node:
                self.dir = dir
            else:
                self.target = self.getNewTarget(self.dir)
            self.setPos()

    def validDirs(self):
        dirs = []
        for key in [UP, DOWN, LEFT, RIGHT]:
            if self.validDir(key):
                if key != self.dir * -1:
                    dirs.append(key)
        if len(dirs) == 0:
            dirs.append(self.dir * -1)
        return dirs

    def randomDir(self, dirs):
        return dirs[self.rng.randint(0, len(dirs) - 1)]

    def goalDir(self, dirs):
        dist = []
        for dir in dirs:
            vec = self.node.pos + self.dirs[dir] * TILE_WIDTH - self.goal
            dist.append(vec.magnitudeSquared())
        index = dist.index(min(dist))
        return dirs[index]

    def setBetweenNodes(self, dir):
        if self.node.neighbors[dir] is not None:
            self.target = self.node.neighbors[dir]
            self.pos = (self.node.pos + self.target.pos) / 2
    
    def reset(self):
        self.setStartNode(self.startNode)
        self.dir = STOP
        self.setSpeed(100)
        self.visible = True