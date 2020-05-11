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
from populations import ESPopulation, CMAPopulation
import skimage
import skimage.io

if __name__ == "__main__":

    env = BoosterBackEnv()
    epds = 8
    pop_size = 64 
    cell_dim = [16,16]

    agent_fn = LSTMAgent
    agent_fn = MLPAgent
    agent_args = dict(obs_dim=49, cell_dim=cell_dim, act_dim=3)
            

    population = CMAPopulation(agent_fn, env, population_size=pop_size, agent_args=agent_args)
    print("policy params: {}".format(population.population[0].num_parameters))


    
    try:
        population.train(generations=1000, epds=epds)
    except KeyboardInterrupt:
        pass
    import pdb; pdb.set_trace();
    #enjoy the results
    env.close()
    env.render = True
    reward_sums = []

    for epd in range(4):
        obs =  env.reset()
        done = False
        sum_rewards = 0.0
        step = 0
        while not done:
            action = population.elite_population[epd % 8]\
                    .get_action(obs)
            obs, reward, done, info = env.step(action)
            time.sleep(0.01)
            sum_rewards += reward
            
            img = p.getCameraImage(512,512)


            skimage.io.imsave("./imgs/epd{}step{}.png".format(epd, step), img[2])

            step += 1
        reward_sums.append(sum_rewards)


    print("average sum of rewards: {:.3e} min: {:.3e} max: {:.3e}"\
            .format(np.mean(reward_sums), np.min(reward_sums), np.max(reward_sums)))


    import pdb; pdb.set_trace();


