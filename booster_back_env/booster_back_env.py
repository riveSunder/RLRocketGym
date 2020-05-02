import time
import os

import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import pybullet as p
import pybullet_data

from agents import RandomAgent, MaxThrustAgent, LSTMAgent

class BoosterBackEnv(gym.Env):

    def __init__(self, render=False, mode="lunar"):
        super(BoosterBackEnv, self).__init__()
        
        self.k_friction = 0.001
        self.render = render
        self.mode = mode
        self.physics_connected = False

        self.max_steps = 1000

    def create_rocket(self, height=1.0, radius=1.):

        path = os.path.abspath(os.path.dirname(__file__))   

        if self.mode == "lunar":
            p.setGravity(0,0.,-1.625)
            self.start_height = 4.0
            self.dry_weight = 1.
            self.fuel = 4.0
            self.max_thrust = [2., 2., 50.]
            
            # not ready yet
            #self.plane_id = p.loadURDF(os.path.join(path,"lunar.urdf"))
            self.plane_id = p.loadURDF("plane.urdf")
            shift = [np.random.randn()*1.0, np.random.randn()*1.0, self.start_height + np.random.random()]

            orientation = p.getQuaternionFromEuler(\
                    [np.random.randn(1) / 5, np.random.randn(1) / 5, np.random.rand(1) / 3])
        else:
            p.setGravity(0, 0, -9.8)
            self.start_height = 4.0
            self.dry_weight = 1.
            self.fuel = 100.0
            self.max_thrust = [200.,200.,5000.]
            self.start_height = 4.
            shift = [np.random.randn()*1.0, np.random.randn()*1.0, self.start_height + np.random.random()]

            orientation = p.getQuaternionFromEuler(\
                    [np.random.randn(1) / 5, np.random.randn(1) / 5, np.random.rand(1) / 3])

            self.plane_id = p.loadURDF("plane.urdf")
        p.setTimeStep(0.01)
        self.step_count = 0


        if self.mode == "lunar":
            self.bot_id = p.loadURDF(os.path.join(path, "lander.xml"), shift, orientation)
        else:
            self.bot_id = p.loadURDF(os.path.join(path, "booster.xml"),\
                shift,\
                orientation)
        #self.create_rocket()
        self.velocity_0 = 0.1
        self.fuel = 100.0
        self.kg_per_kN = 0.03 / 240 # g/(kN*s). 240 is due to the 240 Hz time step in the physics simulatorchangeDynamicsp.change
        p.changeDynamics(self.bot_id, -1, mass = 3.0)
        p.changeDynamics(self.bot_id, 1, mass = 1.0)
        p.changeDynamics(self.bot_id, 0, mass = self.dry_weight + self.fuel)

        # get rid of damping
        num_links = 11
        for ii in range(num_links):
            p.changeDynamics(self.bot_id, ii, angularDamping=0.0, linearDamping=0.0)
        p.changeDynamics(self.bot_id, 0, mass=self.fuel)

        # give the rocket a high incoming velocity 

        p.resetBaseVelocity(self.bot_id, linearVelocity=\
                [np.random.randn(), np.random.randn(), -self.velocity_0 + np.random.randn()*1.0])

    def get_obs(self):

        velocity = p.getBaseVelocity(self.bot_id)
        all_links = [num for num in range(6)]
        link_states = p.getLinkStates(self.bot_id, all_links)
        obs = []

        for kk in range(len(all_links)):
            obs.extend(link_states[kk][0])
            obs.extend(link_states[kk][1])

        obs.extend(velocity[0])
        obs.extend(velocity[1])

        obs.append(self.fuel)
        obs = np.array(obs)

        return obs

    def reset(self):

        if not self.physics_connected:
            # start physics client
            if self.render:
                self.physics_client = p.connect(p.GUI)
            else:
                self.physics_client = p.connect(p.DIRECT)
            self.physics_connected = True

            # add search paths from pybullet for e.g. plane.urdf
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
        else:
            p.resetSimulation()

        self.create_rocket()
        obs = self.get_obs()

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
        
        if 0 * self.render:
            # visual indicators of thrust
            p.changeVisualShape(self.bot_id, -1, rgbaColor=[5e-4*thrust, 0.01, 0.01, 1.0])
            if thrust:
                p.changeVisualShape(self.bot_id, 10, rgbaColor=[\
                        1.0, 0.1, 0.0, thrust/self.max_thrust[2]/2])
            else:
                p.changeVisualShape(self.bot_id, 10, rgbaColor=[0.,0.,0.,0.])

    def apply_control_thrust(self, thrust_x, thrust_y):
        
        if self.fuel > 0.0:
            if (thrust_x + thrust_y) / self.kg_per_kN > self.fuel:
                thrust_x = np.sign(thrust_x) * np.min([self.fuel/self.kg_per_kN/2, np.abs(thrust_x)])
                self.fuel -= np.abs(thrust_x) * self.kg_per_kN
                thrust_y = np.sign(thrust_y) * np.min([self.fuel/self.kg_per_kN/2, np.abs(thrust_y)])
                self.fuel -= np.abs(thrust_y) * self.kg_per_kN

            if self.mode == "lunar":
                p.applyExternalForce(self.bot_id, 2, [thrust_x, thrust_y, 0.0],\
                        posObj=[0.0, 0.0, 2.5], flags=p.LINK_FRAME)
            else:
                p.applyExternalForce(self.bot_id, 1, [thrust_x, thrust_y, 0.0],\
                        posObj=[0.0, 0.0, 5.5], flags=p.LINK_FRAME)

            p.changeDynamics(self.bot_id, 1, mass=self.fuel+self.dry_weight)
            self.fuel = 0.0 if self.fuel <= 0.0 else self.fuel
        else: 
            thrust_x, thrust_y = 0., 0.
            
        if 0 * self.render:
            if thrust_x > 0.:
                p.changeVisualShape(self.bot_id, 7, rgbaColor=[\
                        0.25 + 0.75, 0.1, 0.0, np.abs(thrust_x)/self.max_thrust[0]/2])
                p.changeVisualShape(self.bot_id, 6, rgbaColor=[0.,0.,0.,0.0])
            elif thrust_x < 0.:
                p.changeVisualShape(self.bot_id, 6, rgbaColor=[\
                        0.25 + 0.75, 0.1, 0.0, np.abs(thrust_x)/self.max_thrust[0]/2])
                p.changeVisualShape(self.bot_id, 7, rgbaColor=[0.,0.,0.,0.0])
            else:
                p.changeVisualShape(self.bot_id, 6, rgbaColor=[0.,0.,0.,0.])
                p.changeVisualShape(self.bot_id, 7, rgbaColor=[0.,0.,0.,0.0])

            if thrust_y > 0:
                p.changeVisualShape(self.bot_id, 9, rgbaColor=[\
                        0.25 + 0.75, 0.1, 0.0, np.abs(thrust_y)/self.max_thrust[0]/2])
                p.changeVisualShape(self.bot_id, 8, rgbaColor=[0.,0.,0.,0.0])
            elif thrust_y < 0.:
                p.changeVisualShape(self.bot_id, 8, rgbaColor=[\
                        0.25 + 0.75, 0.1, 0.0, np.abs(thrust_y)/self.max_thrust[0]/2])
                p.changeVisualShape(self.bot_id, 9, rgbaColor=[0.,0.,0.,0.0])
            else:
                p.changeVisualShape(self.bot_id, 8, rgbaColor=[0.,0.,0.,0.])
                p.changeVisualShape(self.bot_id, 9, rgbaColor=[0.,0.,0.,0.0])

    def step(self, action):
    
        sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -709, 709)))
        action[0:2] = np.tanh(action[0:2])
        action[2] = sigmoid(action[2])
        action = [action[mm] * t for mm, t in enumerate(self.max_thrust)]
        
        self.apply_thrust(action[2])

        self.apply_control_thrust(action[0], action[1])

        p.stepSimulation()
        
        obs = self.get_obs()

        reward, info = 0.0, None


        done = False

        # check for collision nose cone on ground plane
        nose_contact_points= p.getContactPoints(self.bot_id, self.plane_id, 1)
        landing_gear_contacts = []
        for link_idx in range(3,7):
            landing_gear_contacts.append(p.getContactPoints(self.bot_id, link_idx))

        landed = 0 not in [len(elem) for elem in landing_gear_contacts]

        if len(nose_contact_points) > 0:
            # print("nose ground collsion")
            done = True
            reward -= 300.0
            # nose cone is on the ground
        elif landed and np.abs(np.mean(obs[-7:-4])) < 1e-2:
            # print("bell is down")
            done = True
            reward += 200.0

        self.step_count += 1
        if self.step_count > self.max_steps:
            done = True

        if done and len(nose_contact_points) == 0:
            reward += self.fuel
        elif not done:
            # survival bonus
            reward += 0.01
        else:
            # lost in space
            reward -= 250
        
        #if done: print(self.step_count)

            
        return obs, reward, done, info

    def close(self):
        if (self.physics_connected):
            p.disconnect()

            self.physics_connected = False




if __name__ == "__main__":

    env = BoosterBackEnv(render=True, mode="booster")
    epds = 8

    #agent = LSTMAgent()
    #agent = MaxThrustAgent()
    agent = RandomAgent()

    for epd in range(epds):
        obs = env.reset()
        done = False
        while not done:

            action = agent.get_action(obs)
            time.sleep(0.01)
            obs, reward, done, info = env.step(action)

    import pdb; pdb.set_trace()
