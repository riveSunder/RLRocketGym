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
