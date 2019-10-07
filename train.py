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
        action = policy_fn()
        rollout_observations.append(observation)
        rollout_actions.append(action)
        observation, reward, done, _ = env.step(action)
        steps += 1
        if render:
            env.render()
        if steps >= max_steps:
            break
        rollout_returns.append(reward)

    return (rollout_observations, rollout_actions, rollout_returns)

def make_random_policy(env):
    np_random = env.np_random
    action_size = len(env.sim.data.ctrl) - 4
    def random_policy():
        random = np_random.uniform(low=-1.0, high=1.0, size=action_size)
        return random
    return random_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_timesteps', type=int)
    args = parser.parse_args()

    env = KokoReacherEnv()

    max_steps = args.max_timesteps or 2000

    random_controller = make_random_policy(env)

    for i in range(10):
        rollout_obs, rollout_act, rollout_r = do_rollout(env, random_controller, max_steps, render=True)
        print("rollout number:", i, " rollout average reward:", sum(rollout_r)/len(rollout_r))

if __name__ == '__main__':
    main()

