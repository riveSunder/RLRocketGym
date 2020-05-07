from abc import ABC, abstractmethod
import copy

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


class DoNothingAgent(Agent):

    def __init__(self):
        pass

    def reset(self):
        pass

    def get_action(self, x):

        return [0.0,\
                0.0,\
                -1000.0]

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

    def sample_parameters(self, pop_mean, covariance, mode="ES"):

        if mode == "ES":
            parameters = np.random.standard_normal(self.num_parameters)

            parameters *= np.diag(covariance)

            parameters += pop_mean
        else:
            # sample using covariance matrix
            parameters = np.random.multivariate_normal(pop_mean, covariance)

        return parameters

    def init_network(self, pop_mean=None, covariance=None, mode="ES"):

        if pop_mean is None:
            pop_mean = np.zeros(self.num_parameters)
        if covariance is None:
            covariance = np.eye(self.num_parameters)

        parameters = self.sample_parameters(pop_mean, covariance, mode=mode)

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
            x = np.clip(x,0,1e5)
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




class MLPAgent(Agent):
    """
    implements a single LSTM agent (numpy only)
    """

    def __init__(self, agent_args):

        # designate dimensions
        self.obs_dim = agent_args["obs_dim"]
        self.cell_dim = agent_args["cell_dim"] #actually hid_dim, but keep name for compatibiilty
        self.act_dim = agent_args["act_dim"]

        # initialize with mean 0.0, variance 1.0
        if type(self.cell_dim) == list:  

            self.num_parameters = self.obs_dim *self.cell_dim[0] \
                    + self.act_dim * self.cell_dim[-1]

            for dim in self.cell_dim[:-1]:
                self.num_parameters += dim**2
                
        elif type(self.cell_dim) == int:
            self.num_parameters = self.obs_dim *self.cell_dim \
                    + self.cell_dim * self.act_dim
            self.cell_dim = [self.cell_dim]
        else:
            assert False, "data type of hidden dimension arg not understood"

        # bias parameters
        self.num_parameters += len(self.cell_dim)
        self.init_network()
        self.reset()

    def sample_parameters(self, pop_mean, covariance, mode="ES"):


        if mode == "ES":
            parameters = np.random.standard_normal(self.num_parameters)

            parameters *= np.diag(covariance)

            parameters += pop_mean
        else:
            # sample using covariance matrix
            if pop_mean.shape[0] != covariance.shape[0]:
                import pdb; pdb.set_trace()
            parameters = np.random.multivariate_normal(pop_mean, covariance)


        return parameters

    def init_network(self, pop_mean=None, covariance=None, mode="ES"):

        if pop_mean is None:
            pop_mean = np.zeros(self.num_parameters)
        if covariance is None:
            covariance = np.eye(self.num_parameters)
        
        parameters = self.sample_parameters(pop_mean, covariance, mode=mode)
        
        self.layers = []
        prev_dim = self.obs_dim * self.cell_dim[0]
        self.layers.append(parameters[:prev_dim].reshape(self.obs_dim, self.cell_dim[0]))

        for oo in range(len(self.cell_dim)-1):
            temp_dim = prev_dim + self.cell_dim[oo] * self.cell_dim[oo+1]
            self.layers.append(parameters[prev_dim:temp_dim]\
                    .reshape(self.cell_dim[oo], self.cell_dim[oo+1]))
            prev_dim = copy.deepcopy(temp_dim)

        temp_dim = prev_dim + self.cell_dim[-1] * self.act_dim

        self.layers.append(parameters[prev_dim:temp_dim]\
                .reshape(self.cell_dim[-1], self.act_dim))

        self.biases = []
        for qq in range(len(self.cell_dim)):
            self.biases.append(parameters[prev_dim+qq])

    def reset(self):
        pass

    def forward(self, x):

        self.sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -709, 709)))

        def relu(x):
            x = np.clip(x,0,1e5)
            return x

        for pp in range(len(self.cell_dim)):
            x = np.tanh(np.matmul(x, self.layers[pp])+self.biases[pp])
            #x = relu(x)

        y = np.matmul(x, self.layers[-1])
        return y
        
    def get_action(self, obs):

        # any sort of pre-processing of obs goes here
        action = self.forward(obs)
        
        # environment is responsible for applying any output activation functions

        return action

    def get_parameters(self):

        parameters = np.array([])
        for param in self.layers:
            parameters = np.concatenate([parameters, param.ravel()])
        for bias in self.biases:
            parameters = np.append(parameters, bias)

        return parameters
