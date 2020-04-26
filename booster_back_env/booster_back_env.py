
import time
import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

class BoosterBackEnv(gym.Env):

    def __init__(self, render=False):
        super(BoosterBackEnv, self).__init__()
        
        # start physics client
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # add search paths from pybullet for e.g. plane.urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -1)
        p.setTimeStep(0.01)
        import pdb; pdb.set_trace()
        plane_ID = p.loadURDF("plane.urdf")

        cube_start_position = [0, 0, 0.001]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))   


        shift = [0, -0.02, 0.0]
        meshScale = [.10, .10, .10]
        self.bot_id = p.loadURDF(os.path.join(path, "booster.xml"),\
            cube_start_position,\
            cube_start_orientation)

        p.changeDynamics(self.bot_id,-1, lateralFriction=self.k_friction)

        p.changeDynamics(self.bot_id,-1, angularDamping=0.1)
        p.changeDynamics(self.bot_id,-1, linearDamping=0.1)


        obs = None

        return obs

    def step(self):
        pass
    


if __name__ == "__main__":

    env = BoosterBackEnv(render=True)
    env.reset()
    import pdb; pdb.set_trace()
