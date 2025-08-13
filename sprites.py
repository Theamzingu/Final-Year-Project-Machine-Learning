import pygame
from constants import *
import numpy as np
from animations import Animate
from copy import deepcopy

BASE_TILE_WIDTH = 16
BASE_TILE_HEIGHT = 16
DEATH = 5
sheet = None

class SpriteSheet:
    def __init__(self):
        global sheet
        if sheet is None:
            sheet = pygame.image.load("spritesheet.png").convert()
            width = int(sheet.get_width() / BASE_TILE_WIDTH * TILE_WIDTH)
            height = int(sheet.get_height() / BASE_TILE_HEIGHT * TILE_HEIGHT)
            sheet = pygame.transform.scale(sheet, (width, height))
        transcolor = sheet.get_at((0, 0))
        sheet.set_colorkey(transcolor)

    def getImage(self, x, y, width, height):
        x *= TILE_WIDTH
        y *= TILE_HEIGHT
        return sheet.subsurface(pygame.Rect(x, y, width, height))
    
class PacmanSprites(SpriteSheet):
    def __init__(self, entity):
        SpriteSheet.__init__(self)
        self.entity = entity
        self.image = self.getStartImage()
        self.animations = {}
        self.defineAnimations()
        self.stopImage = (8, 0)

    def __deepcopy__(self, memo):
        copied = PacmanSprites.__new__(PacmanSprites)
        copied.entity = deepcopy(self.entity, memo)
        copied.image = self.image
        copied.animations = {}
        copied.defineAnimations()
        copied.stopImage = (8, 0)
        return copied

    def getStartImage(self):
        return self.getImage(8, 0)
    
    def getImage(self, x, y):
        return SpriteSheet.getImage(self, x, y, 2 * TILE_WIDTH, 2 * TILE_HEIGHT)
    
    def defineAnimations(self):
        self.animations[LEFT] = Animate(((8, 0), (0, 0), (0, 2), (0, 0)))
        self.animations[RIGHT] = Animate(((10, 0), (2, 0), (2, 2), (2, 0)))
        self.animations[UP] = Animate(((10, 2), (6, 0), (6, 2), (6, 0)))
        self.animations[DOWN] = Animate(((8, 2), (4, 0), (4, 2), (4, 0)))
        self.animations[DEATH] = Animate(((0, 12), (2, 12), (4, 12), (6, 12), (8, 12), (10, 12), (12, 12), (14, 12), (16, 12), (18, 12), (20, 12)), speed = 6, loop = False)

    def update(self, dt):
        if self.entity.alive == True:
            if self.entity.dir == LEFT:
                self.image = self.getImage(*self.animations[LEFT].update(dt))
                self.stopImage = (8, 0)
            elif self.entity.dir == RIGHT:
                self.image = self.getImage(*self.animations[RIGHT].update(dt))
                self.stopImage = (10, 0)
            elif self.entity.dir == UP:
                self.image = self.getImage(*self.animations[UP].update(dt))
                self.stopImage = (10, 2)
            elif self.entity.dir == DOWN:
                self.image = self.getImage(*self.animations[DOWN].update(dt))
                self.stopImage = (8, 2)
            elif self.entity.dir == STOP:
                self.image = self.getImage(*self.stopImage)
        else:
            self.image = self.getImage(*self.animations[DEATH].update(dt))

    def reset(self):
        for key in list(self.animations.keys()):
            self.animations[key].reset()
class GhostSprites(SpriteSheet):
    def __init__(self, entity):
        SpriteSheet.__init__(self)
        self.x = {BLINKY:0, PINKY:2, INKY:4, CLYDE:6}
        self.entity = entity
        self.image = self.getStartImage()

    def __deepcopy__(self, memo):
        copied = GhostSprites.__new__(GhostSprites)
        copied.entity = deepcopy(self.entity, memo)
        copied.x = deepcopy(self.x, memo)
        copied.image = self.image
        return copied

    def getStartImage(self):
        return self.getImage(self.x[self.entity.name], 4)
    
    def getImage(self, x, y):
        return SpriteSheet.getImage(self, x, y, 2 * TILE_WIDTH, 2 * TILE_HEIGHT)

    def update(self, dt):
        x = self.x[self.entity.name]
        if self.entity.mode.current in [SCATTER, CHASE]:
            if self.entity.dir == LEFT:
                self.image = self.getImage(x, 8)
            elif self.entity.dir == RIGHT:
                self.image = self.getImage(x, 10)
            elif self.entity.dir == UP:
                self.image = self.getImage(x, 4)
            elif self.entity.dir == DOWN:
                self.image = self.getImage(x, 6)
        elif self.entity.mode.current == FRIGHT:
            self.image = self.getImage(10, 4)
        elif self.entity.mode.current == SPAWN:
            if self.entity.dir == LEFT:
                self.image = self.getImage(8, 8)
            elif self.entity.dir == RIGHT:
                self.image = self.getImage(8, 10)
            elif self.entity.dir == UP:
                self.image = self.getImage(8, 4)
            elif self.entity.dir == DOWN:
                self.image = self.getImage(8, 6)
class FruitSprites(SpriteSheet):
    def __init__(self, entity, lvl):
        SpriteSheet.__init__(self)
        self.entity = entity
        self.fruits = {0: (16, 8), 1: (18, 8), 2: (20, 8), 3: (16, 10), 4: (18, 10), 5: (20, 10)}
        self.image = self.getStartImage(lvl % len(self.fruits))

    def __deepcopy__(self, memo):
        copied = PacmanSprites.__new__(PacmanSprites)
        copied.entity = deepcopy(self.entity, memo)
        copied.fruits = deepcopy(self.fruits, memo)
        copied.image = self.image
        return copied

    def getStartImage(self, key):
        return self.getImage(*self.fruits[key])
    
    def getImage(self, x, y):
        return SpriteSheet.getImage(self, x, y, 2 * TILE_WIDTH, 2 * TILE_HEIGHT)
    
class LifeSprites(SpriteSheet):
    def __init__(self, numlives):
        SpriteSheet.__init__(self)
        self.resetLives(numlives)

    def removeImage(self):
        if len(self.images) > 0:
            self.images.pop(0)
    
    def resetLives(self, numlives):
        self.images = []
        for i in range(numlives):
            self.images.append(self.getImage(0, 0))
    
    def getImage(self, x, y):
        return SpriteSheet().getImage(x, y, 2 * TILE_WIDTH, 2 * TILE_HEIGHT)
    
class MazeSprites(SpriteSheet):
    def __init__(self, mazefile, rotfile):
        SpriteSheet.__init__(self)
        self.data = self.readMazeFile(mazefile)
        self.rotdata = self.readMazeFile(rotfile)
    
    def getImage(self, x, y):
        return SpriteSheet.getImage(self, x, y, TILE_WIDTH, TILE_HEIGHT)
    
    def readMazeFile(self, mazefile):
        return np.loadtxt(mazefile, dtype = '<U1')
    
    def constructBackground(self, background, y):
        for row in list(range(self.data.shape[0])):
            for col in list(range(self.data.shape[1])):
                if self.data[row][col].isdigit():
                    x = int(self.data[row][col]) + 12
                    sprite = self.getImage(x, y)
                    rotvalue = int(self.rotdata[row][col])
                    sprite = self.rotate(sprite, rotvalue)
                    background.blit(sprite, ((col + 1.5) * TILE_WIDTH, row * TILE_HEIGHT))
                elif self.data[row][col] == '=':
                    sprite = self.getImage(10, 8)
                    background.blit(sprite, ((col + 1.5) * TILE_WIDTH, row * TILE_HEIGHT))
        return background
    
    def rotate(self, sprite, value):
        return pygame.transform.rotate(sprite, value * 90)