
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
        
        self.k_friction = 0.001
        # start physics client
        if render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # add search paths from pybullet for e.g. plane.urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def create_rocket(self, height=1.0, radius=0.1):

        self.bot_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, height=height, radius=radius)


        shift = [0.0, 0.0, 2.0 ]

        orientation = p.getQuaternionFromEuler(\
                [np.random.randn(1)*1e-2,np.random.randn(1)*1e-2,np.random.rand(1)*1e-2])
        
        self.rocket_visual_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                    radius=radius,
                                    length=height,
                                    rgbaColor=[0.1, 0.1, 0.1, 1.0],
                                    specularColor=[0.8, .0, 0],
                                    visualFramePosition=shift, 
                                    visualFrameOrientation=orientation)

        self.rocket_collision_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                radius=radius/2,
                                height=height,
                                collisionFramePosition=shift,
                                collisionFrameOrientation=orientation)

        self.rocket_id = p.createMultiBody(baseMass=10,
                          baseInertialFramePosition=shift,
                          baseVisualShapeIndex=self.rocket_visual_id,
                          baseCollisionShapeIndex=self.rocket_collision_id,
                          basePosition=shift)  

        p.changeDynamics(self.rocket_id, -1, linearDamping=0, angularDamping=0)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -1)
        p.setTimeStep(0.01)
        plane_ID = p.loadURDF("plane.urdf")

        cube_start_position = [0, 0, 0.001]
        cube_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        path = os.path.abspath(os.path.dirname(__file__))   


        shift = [0, -0.02, 0.0]
        meshScale = [.10, .10, .10]
        # self.bot_id = p.loadURDF(os.path.join(path, "booster.xml"),\
        #    cube_start_position,\
        #    cube_start_orientation)
        self.create_rocket()

        p.changeDynamics(self.bot_id,-1, lateralFriction=self.k_friction)

        p.changeDynamics(self.bot_id,-1, angularDamping=0.1)
        p.changeDynamics(self.bot_id,-1, linearDamping=0.1)


        obs = None

        return obs

    def step(self):

        p.stepSimulation()

        obs, reward, info = None, None, None
        done = False
        return obs, reward, done, info

    


if __name__ == "__main__":

    env = BoosterBackEnv(render=True)
    env.reset()

    import pdb; pdb.set_trace()
    for cc in range(1000):
        env.step()
        time.sleep(0.01)
    import pdb; pdb.set_trace()
