import pygame
from vectors import Vector
from constants import *
import numpy as np

class Node:
    def __init__(self, x, y):
        self.pos = Vector(x, y)
        self.neighbors = {UP:None, DOWN:None, LEFT:None, RIGHT:None, PORTAL:None}
        self.access = {UP:[PACMAN, BLINKY, PINK, INKY, CLYDE, FRUIT],
                       DOWN:[PACMAN, BLINKY, PINK, INKY, CLYDE, FRUIT],
                       LEFT:[PACMAN, BLINKY, PINK, INKY, CLYDE, FRUIT],
                       RIGHT:[PACMAN, BLINKY, PINK, INKY, CLYDE, FRUIT]}
        
    def __deepcopy__(self, memo):
        return self
        
    def denyAccess(self, dir, entity):
        if entity.name in self.access[dir]:
            self.access[dir].append(entity.name)

    def allowAccess(self, dir, entity):
        if entity.name not in self.access[dir]:
            self.access[dir].append(entity.name)

    def render(self, screen):
        for n in self.neighbors.keys():
            if self.neighbors[n] is not None:
                line_start = self.pos.asTuple()
                line_end = self.neighbors[n].pos.asTuple()
                pygame.draw.line(screen, WHITE, line_start, line_end, 4)
                pygame.draw.circle(screen, RED, self.pos.asInt(), 12)

class NodeGroup:
    def __init__(self, lvl):
        self.lvl = lvl
        self.nodeList = []
        self.nodeLookUpTable = {}
        self.nodeSymbol = ['+', 'P', 'n']
        self.pathSymbol = ['.', '-', '|', 'p']
        data = self.readMazeFile(lvl)
        self.createNodeTable(data)
        self.connectHor(data)
        self.connectVer(data)
        self.homekey = None

    def __deepcopy__(self, memo):
        return self

    def readMazeFile(self, txtFile):
        return np.loadtxt(txtFile, dtype = '<U1')

    def createNodeTable(self, data, xoffset = 1.5, yoffset = 0):
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in self.nodeSymbol:
                    x, y = self.constructKey(col + xoffset, row + yoffset)
                    new_node = Node(x, y)
                    self.nodeLookUpTable[(x, y)] = new_node
                    self.nodeList.append(new_node)
    
    def constructKey(self, x, y):
        return x * TILE_WIDTH, y * TILE_HEIGHT

    def connectHor(self, data, xoffset = 1.5, yoffset = 0):
        for row in range(data.shape[0]):
            key = None
            for col in range(data.shape[1]):
                if data[row][col] in self.nodeSymbol:
                    if key is None:
                        key = self.constructKey(col + xoffset, row + yoffset)
                    else:
                        otherKey = self.constructKey(col + xoffset, row + yoffset)
                        self.nodeLookUpTable[key].neighbors[RIGHT] = self.nodeLookUpTable[otherKey]
                        self.nodeLookUpTable[otherKey].neighbors[LEFT] = self.nodeLookUpTable[key]
                        key = otherKey
                elif data[row][col] not in self.pathSymbol:
                    key = None
    
    def connectVer(self, data, xoffset = 1.5, yoffset = 0):
        dataTrans = data.transpose()
        for col in range(dataTrans.shape[0]):
            key = None
            for row in range(dataTrans.shape[1]):
                if dataTrans[col][row] in self.nodeSymbol:
                    if key is None:
                        key = self.constructKey(col + xoffset, row + yoffset)
                    else:
                        otherKey = self.constructKey(col + xoffset, row + yoffset)
                        self.nodeLookUpTable[key].neighbors[DOWN] = self.nodeLookUpTable[otherKey]
                        self.nodeLookUpTable[otherKey].neighbors[UP] = self.nodeLookUpTable[key]
                        key = otherKey
                elif dataTrans[col][row] not in self.pathSymbol:
                    key = None

    # def getNodeFromPixles(self, xpix, ypix):
    #     if (xpix, ypix) in self.nodeLookUpTable.keys():
    #         return self.nodeLookUpTable[(xpix, ypix)]
    #     return None
    
    def getNodeFromTile(self, col, row):
        x, y = self.constructKey(col, row)
        if (x, y) in self.nodeLookUpTable.keys():
            return self.nodeLookUpTable[(x, y)]
        return None

    def getStartNode(self):
        node = list(self.nodeLookUpTable.values())
        return node[0]
    
    def setPortalPair(self, pair1, pair2):
        key1 = self.constructKey(*pair1)
        key2 = self.constructKey(*pair2)
        if key1 in self.nodeLookUpTable.keys() and key2 in self.nodeLookUpTable.keys():
            self.nodeLookUpTable[key1].neighbors[PORTAL] = self.nodeLookUpTable[key2]
            self.nodeLookUpTable[key2].neighbors[PORTAL] = self.nodeLookUpTable[key1]
    
    def createHomeNodes(self, xoffset, yoffset):
        homedata = np.array([['X', 'X', '+', 'X', 'X'],
                             ['X', 'X', '.', 'X', 'X'], 
                             ['+', 'X', '+', 'X', '+'], 
                             ['+', '.', '+', '.', '+'],
                             ['+', 'X', 'X', 'X', '+']])
        self.createNodeTable(homedata, xoffset, yoffset)
        self.connectHor(homedata, xoffset, yoffset)
        self.connectVer(homedata, xoffset, yoffset)
        self.homekey = self.constructKey(xoffset + 2, yoffset)
        return self.homekey

    def connectHomeNodes(self, homekey, otherkey, dir):
        key = self.constructKey(*otherkey)
        self.nodeLookUpTable[homekey].neighbors[dir] = self.nodeLookUpTable[key]
        self.nodeLookUpTable[key].neighbors[dir * -1] = self.nodeLookUpTable[homekey]
    
    def denyAccess(self, col, row, dir, entity):
        node = self.getNodeFromTile(col, row)
        if node is not None:
            node.denyAccess(dir, entity)
    
    def allowAccess(self, col, row, dir, entity):
        node = self.getNodeFromTile(col, row)
        if node is not None:
            node.allowAccess(dir, entity)

    def denyAccessList(self, col, row, dir, entities):
        for entity in entities:
            self.denyAccess(col, row, dir, entity)

    def allowAccessList(self, col, row, dir, entities):
        for entity in entities:
            self.allowAccess(col, row, dir, entity)

    def denyHomeAccess(self, entity):
        self.nodeLookUpTable[self.homekey].denyAccess(DOWN, entity)
    
    def allowHomeAccess(self, entity):
        self.nodeLookUpTable[self.homekey].allowAccess(DOWN, entity)

    def denyHomeAccessList(self, entities):
        for entity in entities:
            self.denyHomeAccess(entity)

    def allowHomeAccessList(self, entities):
        for entity in entities:
            self.allowHomeAccess(entity)

    def render(self, screen):
        for node in self.nodeLookUpTable.values():
            node.render(screen)
    