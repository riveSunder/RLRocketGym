from abc import ABC, abstractmethod

import numpy as np

class Agent(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action(self):
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
    def __init__(self, obs_dim=48, act_dim=3, cell_dim=8, population=64):

        # designate dimensions
        self.obs_dim = obs_dim
        self.cell_dim = cell_dim
        self.act_dim = act_dim

        # initialize with mean 0.0, variance 1.0
        self.num_parameters = (cell_dim + obs_dim) * cell_dim * 4 + cell_dim * act_dim
        self.init_network()
        self.reset()

    def init_network(self, pop_mean=None, covariance=None):

        if pop_mean == None:
            print("initializing agent with {} parameters".format(self.num_parameters))
            pop_mean = np.zeros(self.num_parameters)
        if covariance == None:
            covariance = np.eye(self.num_parameters)
        
        parameters = np.random.multivariate_normal(pop_mean, covariance)

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

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

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
        i = relu(np.matmul(x, self.x2i))
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


