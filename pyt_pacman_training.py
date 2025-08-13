from collections import namedtuple, deque
from typing import Optional
from enum import Enum, auto
import random
import math
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import sys
import numpy as np
import torch
import torch.nn as nn
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
from torchrl.envs.utils import check_env_specs, step_mdp

from run import GameController
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
    reward = 0
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
    reward += 1
    reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True).view(*tensordict.shape, 1)
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
    new_dummy_controller = DummyGameController(base_game_controller)
    self.game = new_dummy_controller
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
check_env_specs(env)

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
)
env = env.append_transform(cat_transform)
check_env_specs(env)

start_state = env.reset()
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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity, items=None):
        self.memory = deque(items or [], maxlen=capacity)

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
EPS_DECAY = 2000
TAU = 0.01
LR = 5e-4

# Get number of actions from gym action space
n_actions = 4
# Get the number of state observations
state = env.reset()
n_observations = len(state["observation"])

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(20000)


steps_done = 0

if len(sys.argv) > 1:
    # torch.serialization.add_safe_globals([ReplayMemory, deque, Transition])
    checkpoint = torch.load(sys.argv[1], weights_only=False)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    steps_done = checkpoint['steps_done']
    new_mem_len = len(memory)
    memory = ReplayMemory(new_mem_len, checkpoint['memory'].memory)
    start_episode = checkpoint['episode']
    print(f"Resuming from episode {start_episode}")
else:
    start_episode = 0

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
    num_episodes = 500000
else:
    num_episodes = 50

for i_episode in range(start_episode, num_episodes):
    print("Episode", i_episode)
    total_reward = 0
    # Initialize the environment and get its state
    state = env.reset()
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # print("step", t)
        observation = state["observation"].unsqueeze(0).to(device)
        action = select_action(observation)
        # observation, reward, terminated, truncated, _ = env.step(action.item())
        state["action"] = action
        x = env.step(state)
        reward = x["next", "reward"]
        total_reward += reward
        done = x["next", "done"]
        next_state = step_mdp(x, keep_other=True)
        reward = torch.tensor([reward], device=device)
        # done = terminated or truncated

        # if terminated:
        #     next_state = None
        # else:
        #     next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(observation, action, next_state["observation"].unsqueeze(0).to(device), reward)

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

        if done or t > 10000:
            episode_durations.append(t + 1)
            plot_durations()
            break
    print("total reward", total_reward)
    
    if i_episode % 200 == 0:
        torch.save({
            'episode': i_episode,
            'policy_net_state_dict': policy_net.state_dict(),
            'target_net_state_dict': target_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'steps_done': steps_done,
            'memory': memory,
            }, f"model_training_checkpoints/checkpoint_{i_episode}.tar")

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
