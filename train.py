import argparse
import gym
import numpy as np
import os
import time
from koko_gym import KokoReacherEnv

def do_rollout(env, policy_fn, max_steps, render=False):
    observation = env.reset()
    done = False
    steps = 0

    rollout_observations = []
    rollout_actions = []
    rollout_returns = []

    while not done:
        action = policy_fn(observation[None,:])
        rollout_observations.append(observation)
        rollout_actions.append(action)
        observation, reward, done, _ = env.step(action)
        steps += 1
        if render:
            env.render()
        if steps >= max_steps:
            break

    return (rollout_observations, rollout_actions, rollout_returns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_timesteps', type=int)
    args = parser.parse_args()

    env = KokoReacherEnv()

    max_steps = args.max_timesteps or 1000

    def random_controller(obs):
        # return env.action_space.sample()
        return np.zeros((7, 1))

    for i in range(10):
        do_rollout(env, random_controller, max_steps, render=True)
        print(i)

if __name__ == '__main__':
    main()


import argparse
import gym
import numpy as np
import os
import time
from ant_hilly import AntHillyEnv
from walker2d_hilly import Walker2dHillyEnv
from hopper_hilly import HopperHillyEnv
from mujoco_py import MjViewer

def do_rollout(env, policy_fn, max_steps, render=False):
    observation = env.reset()
    done = False
    steps = 0

    rollout_observations = []
    rollout_actions = []
    rollout_returns = []

    while not done:
        action = policy_fn(observation[None,:])
        rollout_observations.append(observation)
        rollout_actions.append(action)
        observation, reward, done, _ = env.step(action)
        steps += 1
        if render:
            env.render()
        if steps >= max_steps:
            break

    return (rollout_observations, rollout_actions, rollout_returns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--max_timesteps', type=int)
    args = parser.parse_args()

    env = HopperHillyEnv(never_done=True)
    print(env.observation_space.shape)
    exit()

    # env = gym.make(args.envname)

    # max_steps = args.max_timesteps or env.spec.timestep_limit
    max_steps = args.max_timesteps or 1000

    for i in range(10):
        do_rollout(env, lambda obs: env.action_space.sample(), max_steps, render=True)

        if i % 100 == 0:
            print(i)

if __name__ == '__main__':
    main()
