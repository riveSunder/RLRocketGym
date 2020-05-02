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

    #agent_fn = LSTMAgent
    agent_fn = MLPAgent
    agent_args = dict(obs_dim=48, cell_dim=[32,32,32], act_dim=3)
            
    population = ESPopulation(agent_fn, env, population_size=32, agent_args=agent_args)

    try:
        population.train(generations=10, epds=epds)
    except KeyboardInterrupt:
        pass
    
    import pdb; pdb.set_trace();
