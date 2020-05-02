from abc import ABC, abstractmethod

import numpy as np

class Agent(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action(self):
        pass

class Population(ABC):

    @abstractmethod
    def get_episodes(self):
        pass

    @abstractmethod
    def update_population(self):
        pass

    @abstractmethod
    def train(self):
        pass

class ESPopulation(Population):
    """
    Manages the evoluton of a population of agents
    """

    def __init__(self, agent_fn, env, population_size=64, agent_args=None):

        self.env = env
        self.agent_fn = agent_fn
        self.population_size = population_size
        self.agent_args = agent_args

        # evolution in this population has a static parameter variance
        self.var = 3e-1

        self.init_population()
        self.best_agent_performance = -float("Inf")
        self.best_elites_performance = -float("Inf")

    def init_population(self):

        self.population = []

        for ii in range(self.population_size):
            print("initializing agent {} of {}".format(ii+1, self.population_size))
            self.population.append(self.agent_fn(self.agent_args)) 

        self.pop_mean = self.population[0].get_parameters()
        self.covariance = self.var * np.eye(self.population[0].num_parameters)

        self.best_agent = self.population[0]
        self.elite_population = self.population[0]

    def update_population(self):

        for jj in range(self.population_size):

            self.population[jj].init_network(self.pop_mean, self.covariance)

    def get_episodes(self, agent_idx=0, epds=8):

        
        # agent is reset only once, not with every env.reset(), 
        # in case we want to evovle inter-episode meta-learning
        self.population[0].reset()
        fitness = []
        for kk in range(epds):
            obs = self.env.reset()
            done = False
            sum_reward = 0.0
            while not done:
                action = self.population[agent_idx].get_action(obs)
                obs, reward, done, info = self.env.step(action)

                sum_reward += reward

            fitness.append(sum_reward)

        return fitness

    def get_fitness(self, epds=8):

        fitnesses = []
        for kk in range(self.population_size):

            fitnesses.append(np.sum(self.get_episodes(agent_idx=kk, epds=epds))/epds)

        return fitnesses

    def train(self, generations=100, epds=8, verbose=True):

        # elite population is 1/8 of total
        keep = int(0.125 * self.population_size)
        for generation in range(generations):

            fitness = self.get_fitness(epds=epds)

            sort_indices = list(np.argsort(fitness))
            sort_indices.reverse()

            sorted_fitness = np.array(fitness)[sort_indices]

            if self.best_agent_performance < fitness[sort_indices[0]]:

                self.best_agent_performance = fitness[sort_indices[0]]
                self.best_agent = self.population[sort_indices[0]]

            elite_mean = np.mean(sorted_fitness[:keep])
            elite_std = np.std(sorted_fitness[:keep])
            mean_fitness = np.mean(fitness)
            pop_std = np.std(fitness)

            if self.best_elites_performance < elite_mean:

                self.best_elites_performance = elite_mean


            self.elite_population = []
            for mm in range(keep):
                self.elite_population.append(self.population[sort_indices[mm]])

            self.elite_population.append(self.best_agent)

            population_parameters = self.elite_population[0].get_parameters()[np.newaxis,:]

            for ll in range(keep):
                population_parameters = np.concatenate([\
                        population_parameters, self.elite_population[ll].get_parameters()[np.newaxis,:]], axis=0)

            
            self.pop_mean = np.mean(population_parameters, axis=0).squeeze() 
            self.update_population()

            if verbose:
                print("generation {} fitness - elite mean {:.2e}+/- {:2e}, pop mean {:.2e}+/-{:.2e}"\
                       .format(generation, elite_mean, elite_std, mean_fitness, pop_std))
                print("all time best agent/elites fitness {:.2e}/{:.2e}"\
                       .format(self.best_agent_performance, self.best_elites_performance))
