from copy import deepcopy
from constants import *
from fruit import Fruit
from enum import Enum, auto
import heapq
from pygame.locals import *
import random

class UpdateResult(Enum):
    NONE = auto()
    DEAD = auto()
    WON = auto()

class DummyGameController:
    def __init__(self, gamecontroller):
        memo = {}
        self.fruit = deepcopy(gamecontroller.fruit, memo)
        self.lvl = deepcopy(gamecontroller.lvl, memo)
        self.lives = deepcopy(gamecontroller.lives, memo)
        self.score = deepcopy(gamecontroller.score, memo)
        self.fruitCapture = deepcopy(gamecontroller.fruitCapture, memo)
        self.nodes = deepcopy(gamecontroller.nodes, memo)
        self.pacman = deepcopy(gamecontroller.pacman, memo)
        self.pellets = deepcopy(gamecontroller.pellets, memo)
        self.ghost = deepcopy(gamecontroller.ghost, memo)

    def update(self, key_pressed):
        dt = 1.0 / 30.0
        self.pellets.update(dt)
        self.ghost.update(dt)
        if self.fruit is not None:
            self.fruit.update(dt)
        is_game_won = self.checkPelletEvent()
        is_pacman_dead = self.checkGhostEvent()
        self.checkFruitEvent()
        self.pacman.update(dt, key_pressed)
        if is_pacman_dead:
            return UpdateResult.DEAD
        elif is_game_won:
            return UpdateResult.WON
        else:
            return UpdateResult.NONE
        
    def checkFruitEvent(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTile(10.5, 17))
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def checkGhostEvent(self):
        for ghost in self.ghost:
            if self.pacman.collideGhost(ghost):
                # return True
                if ghost.mode.current == FRIGHT:
                    self.updateScore(ghost.points)
                    self.ghost.updatePoints()
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current != SPAWN:
                    return True
        return False

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
        return self.pellets.isEmpty()
    
    def updateScore(self, points):
        self.score += points

def aStar(game_state):
    class QueueEntry:
        def __init__(self, priority, cost, route, controller):
            self.priority = priority
            self.cost = cost
            self.route = route
            self.controller = controller

        def __lt__(self, other):
            return self.priority < other.priority

    start_state = DummyGameController(game_state)
    queue = [QueueEntry(0, 0, [], start_state)]
    # came_from = {}

    def heuristic(state):
        return len(state.pellets.pelletList) * 1000

    TICK_DIVIDER = 30
    state_counter = 0
    successful = False
    while queue:
        state_counter += 1
        if state_counter % 10 == 0:
            print(state_counter)
        entry = heapq.heappop(queue)
        update_result = None
        for key in [K_UP, K_DOWN, K_LEFT, K_RIGHT]:
            key_pressed = {K_UP: False, K_DOWN: False, K_LEFT: False, K_RIGHT: False}
            key_pressed[key] = True
            new_entry = deepcopy(entry)
            last_controller = entry.controller
            for i in range(TICK_DIVIDER):
                update_result = new_entry.controller.update(key_pressed)
                # if i == TICK_DIVIDER - 1:
                #     new_controller = new_entry.controller
                # else:
                #     new_controller = deepcopy(new_entry.controller)
                # came_from[new_controller] = last_controller
                # last_controller = new_controller
                if update_result != UpdateResult.NONE:
                    break
            # print(heuristic(new_state))
            if update_result == UpdateResult.DEAD:
                continue
            elif update_result == UpdateResult.WON:
                successful = True
                for _ in range(TICK_DIVIDER):
                    new_entry.route.append(key)
                break
            new_entry.cost += TICK_DIVIDER
            new_entry.priority = new_entry.cost + heuristic(new_entry.controller)
            for _ in range(TICK_DIVIDER):
                new_entry.route.append(key)
            heapq.heappush(queue, new_entry)
        if update_result == UpdateResult.WON:
            break
    print(f"Success: {successful}")

    # state_route = []
    # current_state = new_controller
    # while current_state is not None:
    #     state_route.append(current_state)
    #     if current_state in came_from:
    #         current_state = came_from[current_state]
    #     else:
    #         current_state = None
    # state_route.reverse()

    return new_entry.route#, state_route
