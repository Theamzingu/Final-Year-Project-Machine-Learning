from collections import defaultdict, namedtuple, deque
from typing import Optional, Dict
from enum import Enum, auto
import random
import math
from copy import deepcopy
from pathlib import Path
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from pygame.locals import *
from constants import *
from fruit import Fruit

from torchrl.data import (
    Binary,
    Bounded,
    Composite,
    Unbounded,
    TensorDictReplayBuffer,
    LazyMemmapStorage,
)
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp

from run import GameController
from vectors import Vector

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

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

base_raw_game_controller = GameController()
base_raw_game_controller.startGame(skip_a_star=True)
base_game_controller = DummyGameController(base_raw_game_controller)

def deserialize_vec(prefix: str, tensordict: TensorDict) -> Vector:
    return Vector(float(tensordict[prefix + "_x"]), float(tensordict[prefix + "_y"]))

def deserialize_to_entity(prefix: str, tensordict: TensorDict, nodes, entity):
    entity.node = nodes.nodeList[int(tensordict[prefix + "_node_i"])]
    entity.target = nodes.nodeList[int(tensordict[prefix + "_target_node_i"])]
    entity.dir = int(tensordict[prefix + "_dir"])
    entity.pos = deserialize_vec(prefix + "_pos", tensordict)
    goal = deserialize_vec(prefix + "_goal", tensordict)
    if goal != Vector(-1000.0, -1000.0):
        entity.goal = goal

def deserialize_pellet_list(prefix: str, tensordict: TensorDict, pellets):
    base_pellet_list = base_game_controller.pellets.pelletList
    for pellet_i in range(len(base_pellet_list)):
        if not bool(tensordict[prefix + "_" + str(pellet_i)]):
            pellets.pelletList.remove(base_pellet_list[pellet_i])

def deserialize_game(tensordict: TensorDict) -> DummyGameController:
    # print("Deserialize:", tensordict)
    new_dummy_controller = DummyGameController(base_game_controller)
    nodes = new_dummy_controller.nodes
    deserialize_to_entity("pacman", tensordict, nodes, new_dummy_controller.pacman)
    deserialize_to_entity("blinky", tensordict, nodes, new_dummy_controller.ghost.blinky)
    deserialize_to_entity("pinky", tensordict, nodes, new_dummy_controller.ghost.pinky)
    deserialize_to_entity("inky", tensordict, nodes, new_dummy_controller.ghost.inky)
    deserialize_to_entity("clyde", tensordict, nodes, new_dummy_controller.ghost.clyde)
    deserialize_pellet_list("pellets", tensordict, new_dummy_controller.pellets)
    return new_dummy_controller

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

def _step(tensordict):
    dummy_controller = deserialize_game(tensordict)
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
    # print("key_i", key_i)
    # print("reward", int(reward))
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
    new_dummy_controller = DummyGameController(base_game_controller)
    serialize_game(tensordict.shape, out, new_dummy_controller)
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
    _step = staticmethod(_step)
    _set_seed = _set_seed

env = PacmanEnv()
check_env_specs(env)

# env = TransformedEnv(
#     env,
#     UnsqueezeTransform(
#         dim = -1,
#         in_keys = ["pacman", "blinky", "pinky", "inky", "clyde", "pellets"],
#         in_keys_inv = ["pacman", "blinky", "pinky", "inky", "clyde", "pellets"],
#     ),
# )

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
check_env_specs(env)

start_state = env.reset()
# print(start_state["observation"])
start_state["action"] = torch.tensor(2, dtype=torch.uint8)
new_state = env.step(start_state)["next"]
if "action" in new_state:
    action = new_state["action"]
else:
    print("action not found in new state, avaiable actions ", new_state.keys())

# NOTE: Start of DQN code

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )
device = "cpu"

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
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

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
state = env.reset()
n_observations = len(state["observation"])

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # rewards = policy_net(state)
            # print(policy_net(state))
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        return torch.tensor([[random.randint(0, 3)]], device=device, dtype=torch.uint8)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    print("Episode", i_episode)
    # Initialize the environment and get its state
    state = env.reset()
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # print("step", t)
        observation = state["observation"].unsqueeze(0)
        action = select_action(observation)
        # observation, reward, terminated, truncated, _ = env.step(action.item())
        state["action"] = action
        x = env.step(state)
        reward = x["next", "reward"]
        done = x["next", "done"]
        next_state = step_mdp(x, keep_other=True)
        reward = torch.tensor([reward], device=device)
        # done = terminated or truncated

        # if terminated:
        #     next_state = None
        # else:
        #     next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(observation, action, next_state["observation"].unsqueeze(0), reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done or t > 500:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()


# NOTE: Start of pendulum tutorial code

# def rollout(steps = 100):
#     data = TensorDict({}, [steps])
#     _data = env.reset()
#     for i in range(steps):
#         _data["action"] = env.action_spec.rand()
#         _data = env.step(_data)
#         data[i] = _data
#         _data = step_mdp(_data, keep_other = True)
#     return data

# print("rollout data ", rollout(100))

# batch_size = 1
# td = env.reset(env.gen_params())
# td = env.rand_step(td)

# rollout = env.rollout(
#     3,
#     auto_reset = False,
#     tensordict = env.reset(env.gen_params()),
# )

# torch.manual_seed(0)
# env.set_seed(0)

# net = nn.Sequential(
#     nn.LazyLinear(128),
#     nn.Tanh(),
#     nn.LazyLinear(128),
#     nn.Tanh(),
#     nn.LazyLinear(128),
#     nn.Tanh(),
#     nn.LazyLinear(1),
# )
# policy = TensorDictModule(
#     net,
#     in_keys = ["observation"],
#     out_keys = ["action"],
# )
# optim = torch.optim.Adam(policy.parameters(), lr =2e-3)

# batch_size = 1
# pbar = tqdm.tqdm(range(20000 // batch_size))
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 20000)
# logs = defaultdict(list)

# for _ in pbar:
#     init_td = env.reset(env.gen_params())
#     # print("init_td keys ", init_td.keys())
#     rollout = env.rollout(100, policy, tensordict = init_td, auto_reset = False)
#     traj_return = rollout["next", "reward"].mean()
#     (-traj_return).backward()
#     gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
#     optim.step()
#     optim.zero_grad()
#     pbar.set_description(
#         f"reward: {traj_return: 44.4f}, "
#         f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, gradient norm: {gn: 4.4f}"
#     )
#     logs["return"].append(traj_return.item())
#     logs["last_reward"].append(rollout[..., -1]["next", "reward"].mean().item())
#     scheduler.step()

# def plot():
#     import matplotlib
#     from matplotlib import pyplot as plt

#     is_ipython = "inline" in matplotlib.get_backend()
#     if is_ipython:
#         from IPython import display
    
#     with plt.ion():
#         plt.figure(figsize = (10, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(logs["return"])
#         plt.title("returns")
#         plt.xlabel("iteration")
#         plt.subplot(1, 2, 2)
#         plt.plot(logs["last_reward"])
#         plt.title("last reward")
#         plt.xlabel("iteration")
#         if is_ipython:
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#         plt.show()
# plot()

# # XXX: Start of Mario tutorial code
#
# class Pacman:
#     def __init__(self, state_dim, action_dim, save_dir):
#         # Act
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.save_dir = save_dir

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#         # Pacman's DNN to predict the most optimal action - we implement this in the Learn section
#         self.net = PacmanNet(self.state_dim, self.action_dim).float()
#         self.net = self.net.to(device=self.device)

#         self.exploration_rate = 1
#         self.exploration_rate_decay = 0.99999975
#         self.exploration_rate_min = 0.1
#         self.curr_step = 0

#         self.save_every = 5e5  # no. of experiences between saving Pacman Net

#         # Cache and Recall
#         self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
#         self.batch_size = 32

#         # TD Estimate and TD Target
#         self.gamma = 0.9
#         self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
#         self.loss_fn = torch.nn.SmoothL1Loss()

#         # Learn
#         self.burnin = 1e4  # min. experiences before training
#         self.learn_every = 3  # no. of experiences between updates to Q_online
#         self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

#     def act(self, state):
#         """
#     Given a state, choose an epsilon-greedy action and update value of step.

#     Inputs:
#     state(``LazyFrame``): A single observation of the current state, dimension is (state_dim)
#     Outputs:
#     ``action_idx`` (``int``): An integer representing which action Pacman will perform
#     """
#         # EXPLORE
#         if np.random.rand() < self.exploration_rate:
#             action_idx = np.random.randint(self.action_dim)

#         # EXPLOIT
#         else:
#             state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
#             state = torch.tensor(state, device=self.device).unsqueeze(0)
#             action_values = self.net(state, model="online")
#             action_idx = torch.argmax(action_values, axis=1).item()

#         # decrease exploration_rate
#         self.exploration_rate *= self.exploration_rate_decay
#         self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

#         # increment step
#         self.curr_step += 1
#         return action_idx

#     def cache(self, state, next_state, action, reward, done):
#         """
#         Store the experience to self.memory (replay buffer)

#         Inputs:
#         state (``LazyFrame``),
#         next_state (``LazyFrame``),
#         action (``int``),
#         reward (``float``),
#         done(``bool``))
#         """
#         # def first_if_tuple(x):
#         #     return x[0] if isinstance(x, tuple) else x
#         # state = first_if_tuple(state).__array__()
#         # next_state = first_if_tuple(next_state).__array__()

#         state = torch.tensor(state)
#         next_state = torch.tensor(next_state)
#         action = torch.tensor([action])
#         reward = torch.tensor([reward])
#         done = torch.tensor([done])

#         # self.memory.append((state, next_state, action, reward, done,))
#         self.memory.add(TensorDict({"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done}, batch_size=[]))

#     def recall(self):
#         """
#         Retrieve a batch of experiences from memory
#         """
#         batch = self.memory.sample(self.batch_size).to(self.device)
#         state, next_state, action, reward, done = (batch.get(key) for key in ("state", "next_state", "action", "reward", "done"))
#         return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

#     def td_estimate(self, state, action):
#         current_Q = self.net(state, model="online")[
#             np.arange(0, self.batch_size), action
#         ]  # Q_online(s,a)
#         return current_Q

#     @torch.no_grad()
#     def td_target(self, reward, next_state, done):
#         next_state_Q = self.net(next_state, model="online")
#         best_action = torch.argmax(next_state_Q, axis=1)
#         next_Q = self.net(next_state, model="target")[
#             np.arange(0, self.batch_size), best_action
#         ]
#         return (reward + (1 - done.float()) * self.gamma * next_Q).float()

#     def update_Q_online(self, td_estimate, td_target):
#         loss = self.loss_fn(td_estimate, td_target)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()

#     def sync_Q_target(self):
#         self.net.target.load_state_dict(self.net.online.state_dict())

#     def save(self):
#         save_path = (
#             self.save_dir / f"pacman_net_{int(self.curr_step // self.save_every)}.chkpt"
#         )
#         torch.save(
#             dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
#             save_path,
#         )
#         print(f"PacmanNet saved to {save_path} at step {self.curr_step}")

#     def learn(self):
#         if self.curr_step % self.sync_every == 0:
#             self.sync_Q_target()

#         if self.curr_step % self.save_every == 0:
#             self.save()

#         if self.curr_step < self.burnin:
#             return None, None

#         if self.curr_step % self.learn_every != 0:
#             return None, None

#         # Sample from memory
#         state, next_state, action, reward, done = self.recall()

#         # Get TD Estimate
#         td_est = self.td_estimate(state, action)

#         # Get TD Target
#         td_tgt = self.td_target(reward, next_state, done)

#         # Backpropagate loss through Q_online
#         loss = self.update_Q_online(td_est, td_tgt)

#         return (td_est.mean().item(), loss)

# class PacmanNet(nn.Module):
#     """mini CNN structure
#   input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
#   """

#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         c, h, w = input_dim

#         # if h != 84:
#         #     raise ValueError(f"Expecting input height: 84, got: {h}")
#         # if w != 84:
#         #     raise ValueError(f"Expecting input width: 84, got: {w}")

#         self.online = self.__build_nn(c, output_dim)

#         self.target = self.__build_nn(c, output_dim)
#         self.target.load_state_dict(self.online.state_dict())

#         # Q_target parameters are frozen.
#         for p in self.target.parameters():
#             p.requires_grad = False

#     def forward(self, input, model):
#         if model == "online":
#             return self.online(input)
#         elif model == "target":
#             return self.target(input)

#     def __build_nn(self, c, output_dim):
#         return nn.Sequential(
#             nn.LazyLinear(128),
#             nn.Tanh(),
#             nn.LazyLinear(128),
#             nn.Tanh(),
#             nn.LazyLinear(128),
#             nn.Tanh(),
#             nn.LazyLinear(1),
#         )
#         # policy = TensorDictModule(
#         #     net,
#         #     in_keys = ["observation"],
#         #     out_keys = ["action"],
#         # )
#         # optim = torch.optim.Adam(policy.parameters(), lr =2e-3)
#         # return nn.Sequential(
#         #     nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
#         #     nn.ReLU(),
#         #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#         #     nn.ReLU(),
#         #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#         #     nn.ReLU(),
#         #     nn.Flatten(),
#         #     nn.Linear(3136, 512),
#         #     nn.ReLU(),
#         #     nn.Linear(512, output_dim),
#         # )

# import time, datetime
# import matplotlib.pyplot as plt

# class MetricLogger:
#     def __init__(self, save_dir):
#         self.save_log = save_dir / "log"
#         with open(self.save_log, "w") as f:
#             f.write(
#                 f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
#                 f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
#                 f"{'TimeDelta':>15}{'Time':>20}\n"
#             )
#         self.ep_rewards_plot = save_dir / "reward_plot.jpg"
#         self.ep_lengths_plot = save_dir / "length_plot.jpg"
#         self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
#         self.ep_avg_qs_plot = save_dir / "q_plot.jpg"

#         # History metrics
#         self.ep_rewards = []
#         self.ep_lengths = []
#         self.ep_avg_losses = []
#         self.ep_avg_qs = []

#         # Moving averages, added for every call to record()
#         self.moving_avg_ep_rewards = []
#         self.moving_avg_ep_lengths = []
#         self.moving_avg_ep_avg_losses = []
#         self.moving_avg_ep_avg_qs = []

#         # Current episode metric
#         self.init_episode()

#         # Timing
#         self.record_time = time.time()

#     def log_step(self, reward, loss, q):
#         self.curr_ep_reward += reward
#         self.curr_ep_length += 1
#         if loss:
#             self.curr_ep_loss += loss
#             self.curr_ep_q += q
#             self.curr_ep_loss_length += 1

#     def log_episode(self):
#         "Mark end of episode"
#         self.ep_rewards.append(self.curr_ep_reward)
#         self.ep_lengths.append(self.curr_ep_length)
#         if self.curr_ep_loss_length == 0:
#             ep_avg_loss = 0
#             ep_avg_q = 0
#         else:
#             ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
#             ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
#         self.ep_avg_losses.append(ep_avg_loss)
#         self.ep_avg_qs.append(ep_avg_q)

#         self.init_episode()

#     def init_episode(self):
#         self.curr_ep_reward = 0.0
#         self.curr_ep_length = 0
#         self.curr_ep_loss = 0.0
#         self.curr_ep_q = 0.0
#         self.curr_ep_loss_length = 0

#     def record(self, episode, epsilon, step):
#         mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
#         mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
#         mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
#         mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
#         self.moving_avg_ep_rewards.append(mean_ep_reward)
#         self.moving_avg_ep_lengths.append(mean_ep_length)
#         self.moving_avg_ep_avg_losses.append(mean_ep_loss)
#         self.moving_avg_ep_avg_qs.append(mean_ep_q)

#         last_record_time = self.record_time
#         self.record_time = time.time()
#         time_since_last_record = np.round(self.record_time - last_record_time, 3)

#         print(
#             f"Episode {episode} - "
#             f"Step {step} - "
#             f"Epsilon {epsilon} - "
#             f"Mean Reward {mean_ep_reward} - "
#             f"Mean Length {mean_ep_length} - "
#             f"Mean Loss {mean_ep_loss} - "
#             f"Mean Q Value {mean_ep_q} - "
#             f"Time Delta {time_since_last_record} - "
#             f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
#         )

#         with open(self.save_log, "a") as f:
#             f.write(
#                 f"{episode:8d}{step:8d}{epsilon:10.3f}"
#                 f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
#                 f"{time_since_last_record:15.3f}"
#                 f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
#             )

#         for metric in ["ep_lengths", "ep_avg_losses", "ep_avg_qs", "ep_rewards"]:
#             plt.clf()
#             plt.plot(getattr(self, f"moving_avg_{metric}"), label=f"moving_avg_{metric}")
#             plt.legend()
#             plt.savefig(getattr(self, f"{metric}_plot"))

# # use_cuda = torch.cuda.is_available()
# use_cuda = False
# print(f"Using CUDA: {use_cuda}")
# print()

# save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
# save_dir.mkdir(parents=True)

# pacman = Pacman(state_dim=(4, 84, 84), action_dim=4, save_dir=save_dir)

# logger = MetricLogger(save_dir)

# # TODO: This needs fixing, consider grabbing learning system from pendulum instead

# episodes = 40
# for e in range(episodes):

#     state = env.reset()

#     # Play the game!
#     while True:

#         # Run agent on the state
#         action = torch.tensor(pacman.act(state), dtype=torch.uint8)
#         state["action"] = action

#         # Agent performs action
#         # next_state, reward, done, trunc, info = env.step(state)
#         x = env.step(state)
#         print(x)
#         reward = x["next", "reward"]
#         done = x["next", "done"]
#         next_state = step_mdp(x, keep_other=True)

#         # Remember
#         pacman.cache(state, next_state, action, reward, done)

#         # Learn
#         q, loss = pacman.learn()

#         # Logging
#         logger.log_step(reward, loss, q)

#         # Update state
#         state = next_state

#         # Check if end of game
#         if done:
#             break

#     logger.log_episode()

#     if (e % 20 == 0) or (e == episodes - 1):
#         logger.record(episode=e, epsilon=pacman.exploration_rate, step=pacman.curr_step)
