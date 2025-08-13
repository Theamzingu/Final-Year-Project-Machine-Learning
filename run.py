import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
from nodes import NodeGroup
from pellets import PelletGroup
from ghost import GhostGroup
from fruit import Fruit
from pause import Pause
from text import TextGroup
from sprites import LifeSprites, MazeSprites
from a_star_ai import aStar
import random
from copy import deepcopy

class GameController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32)
        self.background = None
        self.background_norm = None
        self.background_flash = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.lvl = 0
        self.lives = 5
        self.score = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        self.flashBG = False
        self.flashTime = 0.2
        self.flashTimer = 0
        self.fruitCapture = []

    def nextLvl(self):
        self.showEntities()
        self.lvl += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.lvl)

    def restartGame(self):
        self.lives = 5
        self.lvl = 0
        self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0 
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.lvl)
        self.textgroup.showText(READYTEXT)
        self.lifesprites.resetLives(self.lives)
        self.fruitCapture = []
    
    def resetLvl(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghost.reset()
        self.fruit = None
        self.textgroup.showText(READYTEXT)

    def setBackground(self):
        self.background_norm = pygame.surface.Surface(SCREEN_SIZE).convert()
        self.background_norm.fill(BLACK)
        self.background_flash = pygame.surface.Surface(SCREEN_SIZE).convert()
        self.background_flash.fill(BLACK)
        self.background_norm = self.mazesprites.constructBackground(self.background_norm, self.lvl % 5)
        self.background_flash = self.mazesprites.constructBackground(self.background_flash, 5)
        self.flashBG = False
        self.background = self.background_norm

    def checkEvents(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused = True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETEXT)
                            self.hideEntities()
    
    def render(self):
        self.screen.blit(self.background, (0, 0))
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghost.render(self.screen)
        self.textgroup.render(self.screen)
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREEN_HEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
        pygame.display.update()

    def startGame(self, skip_a_star: bool):
        self.mazesprites = MazeSprites("maze.txt", "maze_rotation.txt")
        self.setBackground()
        self.nodes = NodeGroup("maze.txt")
        self.nodes.setPortalPair((1.5, 14), (28.5, 14))
        homekey = self.nodes.createHomeNodes(13, 11)
        self.nodes.connectHomeNodes(homekey, (13.5, 11), LEFT)
        self.nodes.connectHomeNodes(homekey, (16.5, 11), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTile(16.5, 23))
        self.is_game_starting = True
        self.pellets = PelletGroup("maze.txt")
        self.ghost = GhostGroup(self.nodes.getStartNode(), self.pacman)
        self.ghost.blinky.setStartNode(self.nodes.getNodeFromTile(2 + 13, 0 + 11))
        self.ghost.pinky.setStartNode(self.nodes.getNodeFromTile(2 + 13, 3 + 11))
        self.ghost.inky.setStartNode(self.nodes.getNodeFromTile(0 + 13, 3 + 11))
        self.ghost.clyde.setStartNode(self.nodes.getNodeFromTile(4 + 13, 3 + 11))
        self.ghost.setSpawnNode(self.nodes.getNodeFromTile(2 + 13, 3 + 11))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghost)
        self.nodes.denyAccessList(2 + 13, 3 + 11, LEFT, self.ghost)
        self.nodes.denyAccessList(2 + 13, 3 + 11, RIGHT, self.ghost)
        self.ghost.inky.startNode.denyAccess(RIGHT, self.ghost.inky)
        self.ghost.clyde.startNode.denyAccess(LEFT, self.ghost.clyde)
        self.nodes.denyAccessList(13.5, 11, UP, self.ghost)
        self.nodes.denyAccessList(17.5, 11, UP, self.ghost)
        self.nodes.denyAccessList(13.5, 23, UP, self.ghost)
        self.nodes.denyAccessList(17.5, 23, UP, self.ghost)
        if not skip_a_star:
            # pacman_route, pacman_state_route = aStar(self)
            pacman_route = aStar(self)
            print(list(map(lambda key: pygame.key.name(key), pacman_route)))
            print(len(pacman_route))
            # print(pacman_state_route)
            # self.pacman_route = []
            # for key in pacman_route:
            #     for _ in range(15):
            #         self.pacman_route.append(key)
            self.pacman_route = pacman_route
            self.pacman_route.reverse()
            # self.pacman_state_route = pacman_state_route
            # self.pacman_state_i = 0

    def update(self):
        # dt = self.clock.tick(30) / 1000.0
        self.clock.tick(30)
        dt = 1.0 / 30.0
        self.textgroup.update(dt)
        self.pellets.update(dt)
        is_game_starting = self.is_game_starting
        if not self.pause.paused:
            self.is_game_starting = False
            self.ghost.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvent()
            self.checkGhostEvent()
            self.checkFruitEvent()
        key_pressed = {K_UP: False, K_DOWN: False, K_LEFT: False, K_RIGHT: False}
        if self.pacman_route and not self.pause.paused:
            key_pressed[self.pacman_route.pop()] = True
        else:
            key_pressed = pygame.key.get_pressed()
        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt, key_pressed)
        else:
            self.pacman.update(dt, key_pressed)
        if self.flashBG:
            self.flashBG += dt
            if self.flashTimer >= self.flashTime:
                self.flashTimer = 0
                if self.background == self.background_norm:
                    self.background = self.background_flash
                else:
                    self.background = self.background_norm
        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()
        # dt = self.clock.tick(30) / 1000.0
        # key_pressed = pygame.key.get_pressed()
        # if key_pressed[K_RIGHT]:
        #     self.pacman_state_i = min(self.pacman_state_i + 1, len(self.pacman_state_route) - 1)
        # elif key_pressed[K_LEFT]:
        #     self.pacman_state_i = max(self.pacman_state_i - 1, 0)
        # sim_state = self.pacman_state_route[self.pacman_state_i]
        # self.pacman = sim_state.pacman
        # self.ghost = sim_state.ghost
        # self.pellets = sim_state.pellets
        # self.checkEvents()
        # self.render()

    def checkFruitEvent(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTile(10.5, 17))
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.pos.x, self.fruit.pos.y, 8, time = 1)
                fruitCapture = False
                for fruit in self.fruitCapture:
                    if fruit.get_offset() == self.fruit.image.get_offset():
                        fruitCapture = True
                        break
                if not fruitCapture:
                    self.fruitCapture.append(self.fruit.sprites.image)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def checkGhostEvent(self):
        for ghost in self.ghost:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current == FRIGHT:
                    self.pacman.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.pos.x, ghost.pos.y, 8, time = 1)
                    self.ghost.updatePoints()
                    self.pause.setPause(pauseTime = 1, func = self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current != SPAWN:
                    if self.pacman.alive:
                        self.lives -= 1
                        self.lifesprites.removeImage()
                        self.pacman.dead()
                        self.ghost.hide()
                        if self.lives <= 0:
                            self.textgroup.showText(GAMEOVERTEXT)
                            self.pause.setPause(pauseTime = 3, func = self.restartGame)
                        else:
                            self.pause.setPause(pauseTime = 3, func = self.resetLvl)

    def showEntities(self):
        self.pacman.visible = True
        self.ghost.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghost.hide()

    def checkPelletEvent(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            if self.pellets.numEaten == 30:
                self.ghost.inky.startNode.allowAccess(RIGHT, self.ghost.inky)
            if self.pellets.numEaten == 70:
                self.ghost.clyde.startNode.allowAccess(LEFT, self.ghost.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWER_PELLET:
                self.ghost.startFright()
            if self.pellets.isEmpty():
                self.flashBG = True
                self.hideEntities()
                self.pause.setPause(pauseTime = 3, func = self.nextLvl)
    
    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

if __name__ == "__main__":
    game = GameController()
    game.startGame(skip_a_star=False)
    while True:
        game.update()