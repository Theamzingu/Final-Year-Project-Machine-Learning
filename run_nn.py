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
import random
from copy import deepcopy
from collections import namedtuple, deque
from typing import Optional
from enum import Enum, auto
import random
import math
from copy import deepcopy
from run import GameController

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase
from torch import nn
from pygame.locals import *
from constants import *
from fruit import Fruit

from torchrl.data import (
    Bounded,
    Composite,
    Unbounded,
)
from torchrl.envs import (
    CatTensors,
    EnvBase,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from vectors import Vector

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

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

base_game_controller = GameController()
base_game_controller.startGame(skip_a_star=True)

def serialize_vec(prefix: str, out: TensorDict, batch_size: int, vec: Vector):
    out[prefix + "_x"] = torch.tensor(vec.x, dtype=torch.float32)
    out[prefix + "_y"] = torch.tensor(vec.y, dtype=torch.float32)

def serialize_entity(prefix: str, out: TensorDict, batch_size: int, nodes, entity) -> TensorDict:
    def find_node_i(nodes, node):
        for i, test_node in enumerate(nodes.nodeList):
            if test_node is node:
                return i
    node_i = find_node_i(nodes, entity.node)
    assert node_i is not None
    out[prefix + "_node_i"] = torch.tensor(node_i, dtype=torch.uint8)
    target_node_i = find_node_i(nodes, entity.target)
    assert target_node_i is not None
    out[prefix + "_target_node_i"] = torch.tensor(target_node_i, dtype=torch.uint8)
    out[prefix + "_dir"] = torch.tensor(entity.dir, dtype=torch.int8)
    serialize_vec(prefix + "_pos", out, batch_size, entity.pos)
    serialize_vec(prefix + "_goal", out, batch_size, entity.goal or Vector(-1000.0, -1000.0))

def serialize_pellet_list(prefix: str, out: TensorDict, batch_size, pellets):
    pellets_list = [pellet in pellets.pelletList
        for i, pellet
        in enumerate(base_game_controller.pellets.pelletList)]
    for i, pellet_alive in enumerate(pellets_list):
        # out[prefix + "_" + str(i)] = torch.tensor(pellet_alive, dtype=torch.bool).unsqueeze(-1)
        out[prefix + "_" + str(i)] = torch.tensor(int(pellet_alive), dtype=torch.uint8)

def serialize_game(batch_size: int, tensordict: TensorDict, dummy_controller: DummyGameController):
    nodes = dummy_controller.nodes
    serialize_entity("pacman", tensordict, batch_size, nodes, dummy_controller.pacman)
    serialize_entity("blinky", tensordict, batch_size, nodes, dummy_controller.ghost.blinky)
    serialize_entity("pinky", tensordict, batch_size, nodes, dummy_controller.ghost.pinky)
    serialize_entity("inky", tensordict, batch_size, nodes, dummy_controller.ghost.inky)
    serialize_entity("clyde", tensordict, batch_size, nodes, dummy_controller.ghost.clyde)
    serialize_pellet_list("pellets", tensordict, batch_size, dummy_controller.pellets)

valid_keys = [K_UP, K_DOWN, K_LEFT, K_RIGHT]

def _step(self, tensordict):
    dummy_controller = self.game
    reward = -torch.tensor(0.0, dtype=torch.float32, requires_grad=True).view(*tensordict.shape, 1)
    if dummy_controller.pacman.eatPellets(dummy_controller.pellets.pelletList):
        reward += 10
    for ghost in dummy_controller.ghost:
        if dummy_controller.pacman.collideGhost(ghost) and ghost.mode.current == FRIGHT:
            reward += 200
    done = False
    if dummy_controller.pellets.isEmpty():
        reward += 500
        done = True
    if dummy_controller.checkGhostEvent():
        reward -= 500
        done = True
    reward -= 1
    done = torch.tensor(done, dtype=torch.bool)
    out = TensorDict(
        {
            "params": tensordict["params"],
            "reward": reward,
            "done": done,
        },
        tensordict.shape,
    )
    key_i = min(round(float(tensordict["action"].squeeze(-1))), 3)
    key_pressed = {K_UP: False, K_DOWN: False, K_LEFT: False, K_RIGHT: False}
    key_pressed[valid_keys[key_i]] = True
    dummy_controller.update(key_pressed)
    serialize_game(tensordict.shape, out, dummy_controller)
    return out


def angle_normalize(x):
    return ((x + torch.pi) % (2 * torch.pi)) - torch.pi

def _reset(self, tensordict):
    if tensordict is None or tensordict.is_empty():
        # if no ``tensordict`` is passed, we generate a single set of hyperparameters
        # Otherwise, we assume that the input ``tensordict`` contains all the relevant
        # parameters to get started.
        tensordict = self.gen_params(batch_size=self.batch_size)

    out = TensorDict(
        {
            "params": tensordict["params"],
        },
        batch_size=tensordict.shape,
    )
    global game
    self.game = game
    serialize_game(tensordict.shape, out, game)
    return out

NODE_SPEC = Bounded(
    low=0,
    high=255,
    shape=(),
    dtype=torch.uint8,
)
def make_vec_spec(prefix: str, out):
    out[prefix + "_x"] = Bounded(
        low=-1000.0,
        high=1000.0,
        shape=(),
        dtype=torch.float32,
    )
    out[prefix + "_y"] = Bounded(
        low=-1000.0,
        high=1000.0,
        shape=(),
        dtype=torch.float32,
    )
def make_entity_spec(prefix: str, out):
    make_vec_spec(prefix + "_pos", out)
    out[prefix + "_node_i"] = NODE_SPEC
    out[prefix + "_target_node_i"] = NODE_SPEC
    out[prefix + "_dir"] = Bounded(
        low=-2,
        high=2,
        shape=(),
        dtype=torch.int8,
    )
    make_vec_spec(prefix + "_goal", out)
def make_pellet_list_spec(prefix: str, out):
    for i in range(len(base_game_controller.pellets.pelletList)):
        out[prefix + "_" + str(i)] = Bounded(
            low=0,
            high=1,
            shape=(),
            dtype=torch.uint8,
        )
        # out[prefix + "_" + str(i)] = Binary(shape=(1,), dtype=torch.bool)
spec_fields = {}

def _make_spec(self, td_params):
    make_entity_spec("pacman", spec_fields)
    make_entity_spec("blinky", spec_fields)
    make_entity_spec("pinky", spec_fields)
    make_entity_spec("inky", spec_fields)
    make_entity_spec("clyde", spec_fields)
    make_pellet_list_spec("pellets", spec_fields)
    self.observation_spec = Composite(
        # we need to add the ``params`` to the observation specs, as we want
        # to pass it at each step during a rollout
        params=make_composite_from_td(td_params["params"]),
        shape=(),
        **spec_fields,
    )
    self.state_spec = self.observation_spec.clone()
    # action-spec will be automatically wrapped in input_spec when
    # `self.action_spec = spec` will be called supported
    self.action_spec = Bounded(
        low=0,
        high=3,
        shape=(1,),
        dtype=torch.uint8,
    )
    self.reward_spec = Unbounded(shape=(*td_params.shape, 1))


def make_composite_from_td(td):
    # custom function to convert a ``tensordict`` in a similar spec structure
    # of unbounded values.
    composite = Composite(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else Unbounded(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite

def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng

def gen_params(g=10.0, batch_size=None) -> TensorDictBase:
    """Returns a ``tensordict`` containing the physical parameters such as gravitational force and torque or speed limits."""
    if batch_size is None:
        batch_size = []
    td = TensorDict(
        {
            "params": TensorDict(
                {
                    "max_speed": 8,
                    "max_torque": 2.0,
                    "dt": 0.05,
                    "g": g,
                    "m": 1.0,
                    "l": 1.0,
                },
                [],
            )
        },
        [],
    )
    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td

class PacmanEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(self, td_params=None, seed=None, device="cpu"):
        if td_params is None:
            td_params = self.gen_params()

        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

    # Helpers: _make_step and gen_params
    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec

    # Mandatory methods: _step, _reset and _set_seed
    _reset = _reset
    _step = _step
    _set_seed = _set_seed

env = PacmanEnv()

env = TransformedEnv(
    env,
    # ``Unsqueeze`` the observations that we will concatenate
    UnsqueezeTransform(
        dim=-1,
        in_keys=list(spec_fields.keys()),
        in_keys_inv=list(spec_fields.keys()),
    ),
)

cat_transform = CatTensors(
    in_keys=list(spec_fields.keys()),
    dim=-1,
    out_key="observation",
    del_keys=False,
    # unsqueeze_if_oor=True,
)
env = env.append_transform(cat_transform)
# NOTE: Start of DQN code

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = 4
# Get the number of state observations
n_observations = 279

net = DQN(n_observations, n_actions).to(device)

checkpoint = torch.load(
    'model_training_checkpoints/checkpoint_38000.tar',
    weights_only=False,
    map_location=device,
)
net.load_state_dict(checkpoint['target_net_state_dict'])
net.eval()

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randint(0, 3)]], device=device, dtype=torch.uint8)

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
        self.skip_nn = skip_a_star
        if not skip_a_star:
            global base_game_controller
            base_game_controller = DummyGameController(self)

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
        if not self.skip_nn and not self.pause.paused:
            with torch.no_grad():
                observation = env.reset()["observation"].unsqueeze(0).to(device)
                net_output = net(observation)
                key_i = net_output.max(1).indices.view(1, 1)
                key_i = int(key_i.squeeze(-1))
                key_pressed[valid_keys[key_i]] = True
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