import pygame
from vectors import Vector
from constants import *

class Text:
    def __init__(self, text, color, x, y, size, time = None, id = None, visible = True):
        self.id = id
        self.text = text
        self.color = color
        self.size = size
        self.visible = visible
        self.pos = Vector(x, y)
        self.timer = 0
        self.lifespan = time
        self.label = None
        self.destroy = False
        self.setupFont("PressStart2P-Regular.ttf")
        self.createLabel()

    def setupFont(self, fontpath):
        self.font = pygame.font.Font(fontpath, self.size)

    def createLabel(self):
        self.label = self.font.render(self.text, 1, self.color)

    def setText(self, newText):
        self.text = str(newText)
        self.createLabel()
    
    def update(self, dt):
        if self.lifespan is not None:
            self.timer += dt
            if self.timer >= self.lifespan:
                self.timer = 0
                self.lifespan = None
                self.destroy = True

    def render(self, screen):
        if self.visible:
            x, y = self.pos.asTuple()
            screen.blit(self.label, (x, y))

class TextGroup:
    def __init__(self):
        self.nextId = 10
        self.allText = {}
        self.setupText()
        self.showText(READYTEXT)

    def addText(self, text, color, x, y, size, time = None, id = None):
        self.nextId += 1
        self.allText[self.nextId] = Text(text, color, x, y, size, time, id)
        return self.nextId
    
    def removeText(self, id):
        self.allText.pop(id)
    
    def setupText(self):
        size = TILE_HEIGHT
        self.allText[SCORETEXT] = Text("0".zfill(8), WHITE, 18 * TILE_WIDTH, 33 * TILE_HEIGHT, size)
        self.allText[LEVELTEXT] = Text(str(1).zfill(3), WHITE, 27 * TILE_WIDTH, 33 * TILE_HEIGHT, size)
        self.allText[READYTEXT] = Text("READY!", YELLOW, 12.25 * TILE_WIDTH, 18 * TILE_HEIGHT, size, visible = False)
        self.allText[PAUSETEXT] = Text("PAUSED", YELLOW, 12 * TILE_WIDTH, 18 * TILE_HEIGHT, size, visible = False)
        self.allText[GAMEOVERTEXT] = Text("GAMEOVER" , YELLOW, 11 * TILE_WIDTH, 18 * TILE_HEIGHT, size, visible = False)
        self.addText("SCORE", WHITE, 18 * TILE_WIDTH, 32 * TILE_HEIGHT, size)
        self.addText("LVL", WHITE, 27 * TILE_WIDTH, 32 * TILE_HEIGHT, size)
        
    def update(self, dt):
        for tkey in list(self.allText.keys()):
            self.allText[tkey].update(dt)
            if self.allText[tkey].destroy:
                self.removeText(tkey)
    
    def showText(self, id):
        self.hideText()
        self.allText[id].visible = True

    def hideText(self):
        self.allText[READYTEXT].visible = False
        self.allText[PAUSETEXT].visible = False
        self.allText[GAMEOVERTEXT].visible = False

    def updateLevel(self, level):
        self.updateText(LEVELTEXT, str(level + 1).zfill(3))
    
    def updateScore(self, score):
        self.updateText(SCORETEXT, str(score).zfill(8))

    def updateText(self, id, value):
        if id in self.allText.keys():
            self.allText[id].setText(value)

    def render(self, screen):
        for tkey in list(self.allText.keys()):
            self.allText[tkey].render(screen)