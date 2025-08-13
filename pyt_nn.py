from collections import deque
from constants import *
from copy import deepcopy
from enum import Enum, auto
from fruit import Fruit
from gym.spaces import Box
from gym.wrappers import FrameStack
from pathlib import Path
from PIL import Image
from pygame.locals import *
from tensordict import TensorDict, TensorDictBase
from torch import nn
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    LazyMemmapStorage,
    TensorDictReplayBuffer,
    UnboundedContinuousTensorSpec,
    Binary,
)
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs, step_mdp
from torchvision import transforms as T
from typing import Any, Dict, Optional
from vectors import Vector
from run import GameController
import numpy as np
import random, datetime, os
import torch

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

class PacmanEnv(EnvBase):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    batch_locked = False

    def __init__(
        self,
        base_game_controller: DummyGameController,
        td_params=None,
        seed=None,
        device="cpu",
    ):
        if td_params is None:
            td_params = self.gen_params()
        super().__init__(device=device, batch_size=[])
        self._make_spec(td_params)
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.base_game_controller = base_game_controller

    def _make_spec(self, td_params):
        VEC_SPEC = BoundedTensorSpec(
            low=-1000.0,
            high=1000.0,
            shape=(2,),
            dtype=torch.float32,
        )
        NODE_SPEC = BoundedTensorSpec(
            low=0,
            high=255,
            shape=(),
            dtype=torch.uint8,
        )
        ENTITY_SPEC = CompositeSpec(
            pos=VEC_SPEC,
            node_i=NODE_SPEC,
            target_node_i=NODE_SPEC,
            dir=BoundedTensorSpec(
                low=0,
                high=3,
                shape=(),
                dtype=torch.uint8,
            ),
            goal=VEC_SPEC,
            shape=(),
        )
        PELLET_LIST_SPEC = Binary(shape=(-1,))
        self.observation_spec = CompositeSpec(
            pacman=ENTITY_SPEC,
            blinky=ENTITY_SPEC,
            pinky=ENTITY_SPEC,
            inky=ENTITY_SPEC,
            clyde=ENTITY_SPEC,
            pellets=PELLET_LIST_SPEC,
            # params=make_composite_from_td(td_params["params"]),
            shape=(),
        )
        self.state_spec = self.observation_spec.clone()
        self.action_spec = BoundedTensorSpec(
            low=0,
            high=3,
            shape=(1,),
            dtype=torch.uint8,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*td_params.shape, 1))

    def gen_params(self, batch_size=None) -> TensorDictBase:
        if batch_size is None:
            batch_size = []
        out = TensorDict(
            {
                # "params": TensorDict(
                #     {},
                #     [],
                # )
            },
            [],
        )
        if batch_size:
            out = out.expand(batch_size).contiguous()
        return out

    def _serialize_vec(self, name: str, tensordict: TensorDict, vec: Vector):
        tensordict[name] = torch.tensor([vec.x, vec.y], device=self.device)

    def _serialize_entity(self, batch_size: int, nodes, entity) -> TensorDict:
        out = TensorDict({}, batch_size=batch_size)
        def find_node_i(nodes, node):
            for i, test_node in enumerate(nodes.nodeList):
                if test_node is node:
                    return i
        node_i = find_node_i(nodes, entity.node)
        assert node_i is not None
        out["node_i"] = torch.tensor(node_i, device=self.device)
        target_node_i = find_node_i(nodes, entity.target)
        assert target_node_i is not None
        out["target_node_i"] = torch.tensor(target_node_i, device=self.device)
        out["dir"] = torch.tensor(entity.dir, device=self.device)
        self._serialize_vec("pos", out, entity.pos)
        self._serialize_vec("goal", out, entity.goal or Vector(-1000.0, -1000.0))
        return out

    def _serialize_pellet_list(self, name: str, tensordict: TensorDict, pellets):
        pellets_list = [pellet in pellets.pelletList
            for i, pellet
            in enumerate(self.base_game_controller.pellets.pelletList)]
        tensordict[name] = torch.tensor(pellets_list, device=self.device)

    def _serialize_game(self, batch_size: int, tensordict: TensorDict, dummy_controller: DummyGameController):
        nodes = dummy_controller.nodes
        tensordict["pacman"] = self._serialize_entity(batch_size, nodes, dummy_controller.pacman)
        tensordict["blinky"] = self._serialize_entity(batch_size, nodes, dummy_controller.ghost.blinky)
        tensordict["pinky"] = self._serialize_entity(batch_size, nodes, dummy_controller.ghost.pinky)
        tensordict["inky"] = self._serialize_entity(batch_size, nodes, dummy_controller.ghost.inky)
        tensordict["clyde"] = self._serialize_entity(batch_size, nodes, dummy_controller.ghost.clyde)
        self._serialize_pellet_list("pellets", tensordict, dummy_controller.pellets)

    def _reset(self, tensordict: TensorDict) -> TensorDict:
        if tensordict is None or tensordict.is_empty():
            tensordict = self.gen_params(batch_size=self.batch_size)
        out = TensorDict(
            {
                # "params": tensordict["params"]
            },
            batch_size=tensordict.shape,
        )
        new_dummy_controller = DummyGameController(self.base_game_controller)
        self._serialize_game(tensordict.shape, out, new_dummy_controller)
        print("Reset:", out)
        return out

    def _deserialize_vec(self, tensor) -> Vector:
        return Vector(float(tensor[0]), float(tensor[1]))

    def _deserialize_to_entity(self, tensordict: TensorDict, nodes, entity):
        entity.node = nodes.nodeList[int(tensordict["node_i"])]
        entity.target = nodes.nodeList[int(tensordict["target_node_i"])]
        entity.dir = int(tensordict["dir"])
        entity.pos = self._deserialize_vec(tensordict["pos"])
        goal = self._deserialize_vec(tensordict["goal"])
        if goal != Vector(-1000.0, -1000.0):
            entity.goal = goal

    def _deserialize_pellet_list(self, name: str, tensordict: TensorDict, pellets):
        pellets_list = tensordict[name]
        base_pellet_list = self.base_game_controller.pellets.pelletList
        for pellet_i, pellet_alive in enumerate(pellets_list):
            if not pellet_alive:
                pellets.pelletList.remove(base_pellet_list[pellet_i])

    def _deserialize_game(self, tensordict: TensorDict) -> DummyGameController:
        print("Deserialize:", tensordict)
        new_dummy_controller = DummyGameController(self.base_game_controller)
        nodes = new_dummy_controller.nodes
        self._deserialize_to_entity(tensordict["pacman"], nodes, new_dummy_controller.pacman)
        self._deserialize_to_entity(tensordict["blinky"], nodes, new_dummy_controller.ghost.blinky)
        self._deserialize_to_entity(tensordict["pinky"], nodes, new_dummy_controller.ghost.pinky)
        self._deserialize_to_entity(tensordict["inky"], nodes, new_dummy_controller.ghost.inky)
        self._deserialize_to_entity(tensordict["clyde"], nodes, new_dummy_controller.ghost.clyde)
        self._deserialize_pellet_list("pellets", tensordict, new_dummy_controller.pellets)
        return new_dummy_controller

    def _step(self, tensordict: TensorDict):
        dummy_controller = self._deserialize_game(tensordict)
        reward = -torch.tensor(0.0, device=self.device).view(*tensordict.shape, 1)
        done = torch.zeros_like(reward, dtype=torch.bool)
        if "params" not in tensordict:
            print("No params:", tensordict)
        out = TensorDict(
            {
                # "params": tensordict["params"]
                # if "params" in tensordict
                # else TensorDict({}, batch_size=tensordict.shape),
                "reward": reward,
                "done": done,
            },
            batch_size=tensordict.shape,
        )
        key_pressed = {K_UP: False, K_DOWN: False, K_LEFT: False, K_RIGHT: False}
        dummy_controller.update(key_pressed)
        self._serialize_game(tensordict.shape, tensordict, dummy_controller)
        print("Serialize:", tensordict)
        return out

    # def _step_mdp(self, tensordict: TensorDict) -> TensorDict:
    #     return step_mdp(tensordict, keep_other=True)

    # def simple_rollout(self, steps=100):
    #     # preallocate:
    #     data = TensorDict({}, [steps])
    #     # reset
    #     _data = env.reset()
    #     for i in range(steps):
    #         _data["action"] = env.action_spec.rand()
    #         _data = env.step(_data)
    #         data[i] = _data
    #         _data = step_mdp(_data, keep_other=True)
    #     return data

    def _set_seed(self, seed: Optional[int]):
        pass

game_controller = GameController()
game_controller.startGame(skip_a_star=True)
dummy_game_controller = DummyGameController(game_controller)
env = PacmanEnv(dummy_game_controller)
check_env_specs(env)

# env = JoypadSpace(env, [["right"], ["right", "A"]])
# env.reset()
# next_state, reward, done, trunc, info = env.step(action=0)
# print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
