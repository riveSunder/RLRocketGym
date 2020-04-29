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

class RandomAgent(Agent):

    def __init__(self):
        pass

    def reset(self):
        pass

    def get_action(self, x):

        return [np.random.randn(),\
                np.random.randn(),\
                np.random.random()]

class MaxThrustAgent(Agent):

    def __init__(self):
        pass

    def reset(self):
        pass

    def get_action(self, x):

        return [0.0,\
                0.0,\
                1.0]

class LSTMAgent(Agent):
    """
    implements a single LSTM agent (numpy only)
    """
    def __init__(self, agent_args):

        # designate dimensions
        self.obs_dim = agent_args["obs_dim"]
        self.cell_dim = agent_args["cell_dim"]
        self.act_dim = agent_args["act_dim"]

        # initialize with mean 0.0, variance 1.0
        self.num_parameters = (self.cell_dim + self.obs_dim) * self.cell_dim * 4\
                + self.cell_dim * self.act_dim
        self.init_network()
        self.reset()

    def sample_parameters(self, pop_mean, covariance):

        parameters = np.random.standard_normal(self.num_parameters)

        parameters *= np.diag(covariance)

        parameters += pop_mean

        return parameters

    def init_network(self, pop_mean=None, covariance=None):

        if pop_mean is None:
            pop_mean = np.zeros(self.num_parameters)
        if covariance is None:
            covariance = np.eye(self.num_parameters)
        
        parameters = self.sample_parameters(pop_mean, covariance)

        dim_x2f = (self.obs_dim + self.cell_dim) * self.cell_dim
        dim_c2y = dim_x2f*4 + self.cell_dim * self.act_dim
        
        self.x2f = parameters[dim_x2f*0:dim_x2f*1].reshape(self.obs_dim+self.cell_dim, self.cell_dim)
        self.x2i = parameters[dim_x2f*1:dim_x2f*2].reshape(self.obs_dim+self.cell_dim, self.cell_dim)
        self.x2j = parameters[dim_x2f*2:dim_x2f*3].reshape(self.obs_dim+self.cell_dim, self.cell_dim)
        self.x2o = parameters[dim_x2f*3:dim_x2f*4].reshape(self.obs_dim+self.cell_dim, self.cell_dim)

        self.c2y = parameters[dim_x2f*4:dim_c2y].reshape(self.cell_dim, self.act_dim)

        # Maybe later set this up for a cell state with dimensions n by self.cell_dim, 
        # for parallelization
        self.cell_state = np.zeros(self.cell_dim)

    def reset(self):
        self.cell_state *= 0.0

    def forward(self, x):

        self.sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -709, 709)))

        def relu(x):
            x[x<0.0] *= 0.0
            return x

        if 1:
            # default gates are open
            bias_f = 2.0
            

        x = np.concatenate([x, self.cell_state])
        # forget gate
        f = self.sigmoid(np.matmul(x, self.x2f) + bias_f)
        # input layer
        i = np.tanh(np.matmul(x, self.x2i))
        # input gate
        j = self.sigmoid(np.matmul(x, self.x2j) + bias_f)
        # output gate
        o = self.sigmoid(np.matmul(x, self.x2o) + bias_f)

        self.cell_state = (self.cell_state * f) + (i * j)
        
        # output
        y = np.matmul(self.cell_state, self.c2y)

        return y
        
    def get_action(self, obs):

        # any sort of pre-processing of obs goes here
        action = self.forward(obs)
        
        # control thrust (actions 0 and 1) are squashed between +/- 1.0
        action[0:2] = np.tanh(action[0:2])

        # the main thruster (action 2) can only be positive
        action[2] = self.sigmoid(action[2])

        return action

    def get_parameters(self):

        parameters = np.array([])
        for param in [self.x2f, self.x2j, self.x2i, self.x2o, self.c2y]:

            parameters = np.concatenate([parameters, param.ravel()])

        return parameters

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
        self.var = 1e-1

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


