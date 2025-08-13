import pygame
from pygame.locals import *
from vectors import Vector
from constants import *
from entity import Entity
from modes import ModeController
from sprites import GhostSprites

class Ghost(Entity):
    def __init__(self, node, pacman = None, blinky = None):
        Entity.__init__(self, node)
        self.name = GHOST
        self.points = 200
        self.goal = Vector()
        self.dirMethod = self.goalDir
        self.pacman = pacman
        self.mode = ModeController(self)
        self.blinky = blinky
        self.homeNode = node

    def update(self, dt):
        self.sprites.update(dt)
        self.mode.update(dt)
        if self.mode.current == SCATTER:
            self.scatter()
        elif self.mode.current == CHASE:
            self.chase()
        Entity.update(self, dt)

    def scatter(self):
        self.goal = Vector()

    def chase(self):
        self.goal = self.pacman.pos

    def startFright(self):
        self.mode.setFrightMode()
        self.setSpeed(50)
        self.dirMethod = self.randomDir
    
    def normalMode(self):
        self.setSpeed(100)
        self.dirMethod = self.goalDir
        self.homeNode.denyAccess(DOWN, self)
    
    def spawn(self):
        self.goal = self.spawnNode.pos
    
    def setSpawnNode(self, node):
        self.spawnNode = node

    def startSpawn(self):
        self.mode.setSpawnMode()
        if self.mode.current == SPAWN:
            self.setSpeed(150)
            self.dirMethod = self.goalDir
            self.spawn()
    
    def reset(self):
        Entity.reset(self)
        self.points = 200
        self.dirMethod = self.goalDir

class Blinky(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = BLINKY
        self.color = RED
        self.sprites = GhostSprites(self)

class Pinky(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = PINKY
        self.color = PINK
        self.sprites = GhostSprites(self)
    
    def scatter(self):
        self.goal = Vector(TILE_WIDTH * N_COLS, 0)
    
    def chase(self):
        self.goal = self.pacman.pos + self.pacman.dirs[self.pacman.dir] * TILE_WIDTH * 4

class Inky(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = INKY
        self.color = TEAL
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector(TILE_WIDTH * N_COLS, TILE_HEIGHT * N_ROWS)
    
    def chase(self):
        vec1 = self.pacman.pos + self.pacman.dirs[self.pacman.dir] * TILE_WIDTH * 2
        vec2 = (vec1 - self.blinky.pos) * 2
        self.goal = self.blinky.pos + vec2

class Clyde(Ghost):
    def __init__(self, node, pacman = None, blinky = None):
        Ghost.__init__(self, node, pacman, blinky)
        self.name = CLYDE
        self.color = ORANGE
        self.sprites = GhostSprites(self)

    def scatter(self):
        self.goal = Vector(0, TILE_HEIGHT * N_ROWS)

    def chase(self):
        d = self.pacman.pos - self.pos
        ds = d.magnitudeSquared()
        if ds <= (TILE_WIDTH * 8) ** 2:
            self.scatter()
        else:
            self.goal = self.pacman.pos + self.pacman.dirs[self.pacman.dir] * TILE_WIDTH * 4

class GhostGroup:
    def __init__(self, node, pacman):
        self.blinky = Blinky(node, pacman)
        self.pinky = Pinky(node, pacman)
        self.inky = Inky(node, pacman, self.blinky)
        self.clyde = Clyde(node, pacman)
        self.ghosts = [self.blinky, self.pinky, self.inky, self.clyde]

    def __iter__(self):
        return iter(self.ghosts)

    def update(self, dt):
        for ghost in self:
            ghost.update(dt)
    
    def startFright(self):
        for ghost in self:
            ghost.startFright()
        self.resetPoints()

    def setSpawnNode(self, node):
        for ghost in self:
            ghost.setSpawnNode(node)

    def updatePoints(self):
        for ghost in self:
            ghost.points *= 2
    
    def resetPoints(self):
        for ghost in self:
            ghost.points = 200

    def reset(self):
        for ghost in self:
            ghost.reset()

    def hide(self):
        for ghost in self:
            ghost.visible = False
    
    def show(self):
        for ghost in self:
            ghost.visible = True
    
    def render(self, screen):
        for ghost in self:
            ghost.render(screen)