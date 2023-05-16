import warnings

warnings.filterwarnings('ignore')

from pyvirtualdisplay import Display

Display(visible=False, size=(1400, 900)).start()

import copy
import torch
import random
import gym
import matplotlib
import functools
import itertools
import math

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from collections import deque, namedtuple
from IPython.display import HTML
from base64 import b64encode

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from torch.distributions import Normal

from pytorch_lightning import LightningModule, Trainer

import brax
from brax import envs
from brax.envs import to_torch
from brax.io import html

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

v = torch.ones(1, device='cuda')


@torch.no_grad()
def create_video(env, episode_length, policy=None):
    qp_array = []
    state = env.reset()
    for i in range(episode_length):
        if policy:
            loc, scale = policy(state)
            sample = torch.normal(loc, scale)
            action = torch.tanh(sample)
        else:
            action = env.action_space.sample()
        state, _, _, _ = env.step(action)
        qp_array.append(env.unwrapped._state.qp)
    return HTML(html.render(env.unwrapped._env.sys, qp_array))


@torch.no_grad()
def test_agent(env, episode_length, policy, episodes=10):
    ep_returns = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            loc, scale = policy(state)
            sample = torch.normal(loc, scale)
            action = torch.tanh(sample)
            state, reward, done, info = env.step(action)
            ep_ret += reward.item()

        ep_returns.append(ep_ret)

    return sum(ep_returns) / episodes


# Create the policy
class GradientPolicy(nn.Module):

    def __init__(self, in_features, out_dims, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, out_dims)
        self.fc_std = nn.Linear(hidden_size, out_dims)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        loc = self.fc_mu(x)
        loc = torch.tanh(loc)
        scale = self.fc_std(x)
        scale = F.softplus(scale) + 0.001
        return loc, scale


# Create the value network
class ValueNet(nn.Module):

    def __init__(self, in_features, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float32).to(device)
        self.var = torch.ones(shape, dtype=torch.float32).to(device)
        self.count = epsilon

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):

    def __init__(self, env, epsilon=1e-8):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape[-1])
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        obs = self.normalize(obs)
        return obs, rews, dones, infos

    def reset(self, **kwargs):
        return_info = kwargs.get("return_info", False)
        if return_info:
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
        obs = self.normalize(obs)
        if not return_info:
            return obs
        else:
            return obs, info

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)


def create_env(env_name, num_envs=256, episode_length=1000):
    env = gym.make(env_name, batch_size=num_envs, episode_length=episode_length)
    env = to_torch.JaxToTorchWrapper(env, device=device)
    env = NormalizeObservation(env)
    return env


class RLDataset(IterableDataset):
    def __init__(self, env, policy, samples_per_epoch, epoch_repeat):
        self.env = env
        self.policy = policy
        self.samples_per_epoch = samples_per_epoch
        self.epoch_repeat = epoch_repeat
        self.obs = self.env.reset()

    @torch.no_grad()
    def __iter__(self):
        transitions = []
        for step in range(self.samples_per_epoch):
            loc, scale = self.policy(self.obs)
            action = torch.normal(loc, scale)
            next_obs, reward, done, info = self.env.step(action)
            transitions.append((self.obs, loc, scale, reward, done, next_obs))
            self.obs = next_obs

        num_samples = self.env.num_envs * self.samples_per_epoch
        reshape_fn = lambda x: x.view(num_samples, -1)
        batch = map(torch.stack, zip(*transitions))
        """
        transform the tensor:
        (num_samples, num_envs, feature_dims) -> (num_samples * num_envs, feature_dims)
        """
        obs_b, loc_b, scale_b, action_b, reward_b, done_b, next_obs_b = \
            map(reshape_fn, batch)

        for repeat in range(self.epoch_repeat):
            idx = list(range(num_samples))
            random.shuffle(idx)

            for i in idx:
                yield obs_b[i], loc_b[i], scale_b[i], action_b[i], \
                    reward_b[i], done_b[i], next_obs_b[i]


class PPO(LightningModule):

    def __init__(self, env_name, num_envs=2048, episode_length=1000,
                 batch_size=1024, hidden_size=256, samples_per_epoch=5,
                 epoch_repeat=8, policy_lr=1e-4, value_lr=1e-3, gamma=0.97,
                 epsilon=0.3, entropy_coef=0.1, optim=AdamW):
        super().__init__()
        self.env = create_env(env_name, num_envs=num_envs, episode_length=episode_length)
        test_env = gym.make(env_name, episode_length=episode_length)
        test_env = to_torch.JaxToTorchWrapper(test_env, device=device)
        self.test_env = NormalizeObservation(test_env)
        self.test_env.obs_rms = self.env.obs_rms

        obs_size = self.env.observation_space.shape[1]
        action_dims = self.env.action_space.shape[1]

        self.policy = GradientPolicy(obs_size, action_dims, hidden_size)
        self.value_net = ValueNet(obs_size, hidden_size)
        self.target_value_net = copy.deepcopy(self.value_net)

        self.dataset = RLDataset(self.env, self.policy, samples_per_epoch, epoch_repeat)
        self.save_hyperparameters()
        self.videos = []

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        return value_opt, policy_opt

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)


def main():
    entry_point = functools.partial(envs.create_gym_env, env_name='ant')
    gym.register('brax-ant-v0', entry_point=entry_point)
    # env = gym.make('brax-ant-v0', episode_length=1000)
    # env = to_torch.JaxToTorchWrapper(env, device=device)
    # create_video(env, 1000)

    env = create_env('brax-ant-v0', num_envs=10)
    # obs = env.reset()
    # print("Num envs: ", obs.shape[0], "Obs dimensions: ", obs.shape[1])
    # print("Action space: ", env.action_space)
    # obs, reward, done, info = env.step(env.action_space.sample())
    # print("Observation: ", obs)
    # print("\n\n======\n\n")
    # print("Reward: ", reward)
    # print("\n\n======\n\n")
    # print("Done: ", done)
    # print("\n\n======\n\n")
    # print("Info: ", info)


if __name__ == "__main__":
    main()
