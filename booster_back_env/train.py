import time
import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

from agents import RandomAgent, MaxThrustAgent, LSTMAgent, MLPAgent
from booster_back_env import BoosterBackEnv
from populations import ESPopulation

if __name__ == "__main__":

    env = BoosterBackEnv()
    epds = 4 
    pop_size = 128

    #agent_fn = LSTMAgent
    agent_fn = MLPAgent
    agent_args = dict(obs_dim=49, cell_dim=[32,32], act_dim=3)
            

    population = ESPopulation(agent_fn, env, population_size=pop_size, agent_args=agent_args)

    try:
        population.train(generations=1000, epds=epds)
    except KeyboardInterrupt:
        pass
    import pdb; pdb.set_trace();
    #enjoy the results
    env.close()
    env.render = True
    reward_sums = []
    for epd in range(10):
        obs =  env.reset()
        done = False
        sum_rewards = 0.0
        while not done:
            action = population.best_agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(0.05)
            sum_rewards += reward

        reward_sums.append(sum_rewards)

    print("average sum of rewards: {:.3e} min: {:.3e} max: {:.3e}"\
            .format(np.mean(reward_sums), np.min(reward_sums), np.max(reward_sums)))


    import pdb; pdb.set_trace();


