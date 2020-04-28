
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

    def create_rocket(self, height=1.0, radius=1.):
        pass

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.01)
        self.plane_id = p.loadURDF("plane.urdf")
        self.dry_weight = 20.

        path = os.path.abspath(os.path.dirname(__file__))   

        shift = [0.,0.,2.] #[np.random.randn()*10, np.random.randn()*10, .30 + np.random.random()*.20]

        orientation = p.getQuaternionFromEuler(\
                [np.random.randn(1)*1e-1,np.random.randn(1)*1e-1,np.random.rand(1)*1e-1])
        self.bot_id = p.loadURDF(os.path.join(path, "booster.xml"),\
            shift,\
            orientation)
        #self.create_rocket()



        self.fuel = 100.0
        self.kg_per_kN = 0.3 / 240 # g/(kN*s). 240 is due to the 240 Hz time step in the physics simulatorchangeDynamicsp.change
        p.changeDynamics(self.bot_id,-1, mass = 3.0)
        p.changeDynamics(self.bot_id, 2, mass = 1.0)
        p.changeDynamics(self.bot_id, 1, mass = self.dry_weight + self.fuel)

        # get rid of damping
        num_links = 7
        for ii in range(num_links):
            p.changeDynamics(self.bot_id, ii, angularDamping=0.0, linearDamping=0.0)
        p.changeDynamics(self.bot_id, 1, mass=self.fuel)
        

        # give the rocket a random push

        p.applyExternalForce(self.bot_id, 1, np.random.randn(3),[0,0,0],flags=p.LINK_FRAME)
        obs = None

        return obs

    def apply_torque(self,torque):

        p.setJointMotorControl2(bodyIndex=self.bot_id, jointIndex=0,\
                controlMode = p.TORQUE_CONTROL, force=torque)
                
    def apply_thrust(self, thrust):

        if self.fuel > 0.0:
            thrust = np.min([self.fuel / self.kg_per_kN, thrust])
            # apply thrust to the axis of the rocket's bell nozzle
            p.applyExternalForce(self.bot_id, -1, [0.0, 0.0, thrust],\
                    posObj=[0,0,1.0],flags=p.LINK_FRAME)

            # decrement fuel proportional to the thrust used
            self.fuel -= thrust * self.kg_per_kN

            p.changeDynamics(self.bot_id, 1, mass=self.fuel+self.dry_weight)
            self.fuel = 0.0 if self.fuel <= 0.0 else self.fuel
        else:
            thrust = 0.0
        
        # visual indicators of thrust
        p.changeVisualShape(self.bot_id, -1, rgbaColor=[5e-4*thrust, 0.01, 0.01, 1.0])
        if thrust:
            p.changeVisualShape(self.bot_id, 10, rgbaColor=[\
                    0.75 + 0.25*np.random.random(), 0.1, 0.0, 0.35 + np.random.random()/2])
        else:
            p.changeVisualShape(self.bot_id, 10, rgbaColor=[0.,0.,0.,0.])

    def apply_control_thrust(self, thrust_x, thrust_y):
        
        if self.fuel > 0.0:
            if (thrust_x + thrust_y) / self.kg_per_kN > self.fuel:
                thrust_x = np.sign(thrust_x) * np.min([self.fuel/self.kg_per_kN/2, np.abs(thrust_x)])
                self.fuel -= np.abs(thrust_x) * self.kg_per_kN
                thrust_y = np.sign(thrust_y) * np.min([self.fuel/self.kg_per_kN/2, np.abs(thrust_y)])
                self.fuel -= np.abs(thrust_y) * self.kg_per_kN

            p.applyExternalForce(self.bot_id, 1, [thrust_x, thrust_y, 0.0],\
                    posObj=[0.0, 0.0, 5.5], flags=p.LINK_FRAME)

            p.changeDynamics(self.bot_id, 1, mass=self.fuel+self.dry_weight)
            self.fuel = 0.0 if self.fuel <= 0.0 else self.fuel
        else: 
            thrust_x, thrust_y = 0., 0.
            
        if thrust_x > 0.:
            p.changeVisualShape(self.bot_id, 7, rgbaColor=[\
                    np.random.random()*0.25 + 0.75, 0.1, 0.0, 0.4 + np.random.random()/2])
            p.changeVisualShape(self.bot_id, 6, rgbaColor=[0.,0.,0.,0.0])
        elif thrust_x < 0.:
            p.changeVisualShape(self.bot_id, 6, rgbaColor=[\
                    np.random.random()*0.25 + 0.75, 0.1, 0.0, 0.4 + np.random.random()/2])
            p.changeVisualShape(self.bot_id, 7, rgbaColor=[0.,0.,0.,0.0])
        else:
            p.changeVisualShape(self.bot_id, 6, rgbaColor=[0.,0.,0.,0.])
            p.changeVisualShape(self.bot_id, 7, rgbaColor=[0.,0.,0.,0.0])

        if thrust_y > 0:
            p.changeVisualShape(self.bot_id, 9, rgbaColor=[\
                    np.random.random()*0.25 + 0.75, 0.1, 0.0, 0.4 +np.random.random()/2])
            p.changeVisualShape(self.bot_id, 8, rgbaColor=[0.,0.,0.,0.0])
        elif thrust_y < 0.:
            p.changeVisualShape(self.bot_id, 8, rgbaColor=[\
                    np.random.random()*0.25 + 0.75, 0.1, 0.0, 0.4 + np.random.random()/2])
            p.changeVisualShape(self.bot_id, 9, rgbaColor=[0.,0.,0.,0.0])
        else:
            p.changeVisualShape(self.bot_id, 8, rgbaColor=[0.,0.,0.,0.])
            p.changeVisualShape(self.bot_id, 9, rgbaColor=[0.,0.,0.,0.0])

    def step(self, action):
        
        self.apply_thrust(action[2])


        self.apply_control_thrust(action[0], action[1])

        p.stepSimulation()
        
        velocity = p.getBaseVelocity(self.bot_id)
        nosecone_state = p.getLinkState(self.bot_id, 4)
        bell_nozzle_state = p.getLinkState(self.bot_id,0)
        
        obs, reward, info = None, 0.0, None

        done = False

        # check for collision nose cone on ground plane
        nose_contact_points= p.getContactPoints(self.bot_id, self.plane_id, 1)

        if len(nose_contact_points) > 0:
            print("nose ground collsion")
            done = True
            reward -= 300.0
            # nose cone is on the ground
        elif np.abs(np.mean(velocity[0])) < 1e-4:
            print("bell is down")
            done = True
            reward += 100.0

        
        if done:
            reward += self.fuel
            
        return obs, reward, done, info

    


if __name__ == "__main__":

    env = BoosterBackEnv(render=True)
    env.reset()

    done = False
    while not done:

        action = [1 * np.random.randn(), 1. *np.random.randn(), 5.0] 
        obs, reward, done, info = env.step(action)
        time.sleep(0.001)

    import pdb; pdb.set_trace()
