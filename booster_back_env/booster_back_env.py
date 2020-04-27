
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

        self.bot_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, height=height, radius=radius)


        shift = [0.0, 0.0, 1.0 ]
        orientation = p.getQuaternionFromEuler(\
                [np.random.randn(1)*1e-1,np.random.randn(1)*1e-1,np.random.rand(1)*1e-1])
        path = os.path.abspath(os.path.dirname(__file__))   
        mesh_scale = [1.0,1.0,1.0]
        shift_body = [0.0,0.0,1.1]
        shift_nosecone = [0.0,0.0,height+0.1]
        

        self.body_visual_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER,
                                    radius=radius,
                                    length=height,
                                    rgbaColor=[0.5, 0.5, 0.65, 1.0],
                                    specularColor=[0.8, .0, 0],
                                    visualFramePosition=shift_body) 
        self.body_collision_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER,
                                radius=radius/2,
                                height=height,
                                collisionFramePosition=shift_body)
        self.nozzle_visual_id= p.createVisualShape(shapeType=p.GEOM_MESH,
                                fileName=os.path.join(path,"bell_nozzle.stl"),
                                rgbaColor=[0.1,0.1,0.1,1.0],
                                meshScale=mesh_scale)
        self.nozzle_collision_id= p.createCollisionShape(shapeType=p.GEOM_MESH,
                                fileName=os.path.join(path,"bell_nozzle.stl"),
                                meshScale= mesh_scale)
        self.nosecone_visual_id= p.createVisualShape(shapeType=p.GEOM_MESH,
                                fileName=os.path.join(path,"nose_cone.stl"),
                                rgbaColor=[0.5,0.5,0.75,1.0],
                                visualFramePosition=shift_nosecone,
                                meshScale=mesh_scale)
        self.nosecone_collision_id= p.createCollisionShape(shapeType=p.GEOM_MESH,
                                fileName=os.path.join(path,"nose_cone.stl"),
                                collisionFramePosition=shift_nosecone,
                                meshScale= mesh_scale)

#        temp = p.createMultiBody(baseMass=10,
#                baseInertialFramePosition=shift_body,
#                baseInertialFrameOrientation=orientation,
#                baseVisualShapeIndex=self.body_visual_id,
#                baseCollisionShapeIndex=self.body_collision_id,
#                basePosition=shift_body)
        self.rocket_id = p.createMultiBody(baseMass=10,
                          baseInertialFramePosition=shift,
                          baseInertialFrameOrientation=orientation,
                          baseVisualShapeIndex=self.nozzle_visual_id,
                          baseCollisionShapeIndex=self.nozzle_collision_id,
                          linkVisualShapeIndices=[self.nosecone_visual_id],
                          linkCollisionShapeIndices=[self.nosecone_collision_id],
                          linkMasses = [10.0],
                          linkPositions=[[0,0,10]],
                          linkOrientations=[orientation],
                          linkInertialFramePositions=[[0,0,10]],
                          linkInertialFrameOrientations=[orientation],
                          linkJointAxis=[0],
                          linkParentIndices=[0],
                          linkJointTypes=[p.JOINT_REVOLUTE],
                          basePosition=shift)  

        p.changeDynamics(self.rocket_id, -1, linearDamping=0, angularDamping=0)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(0.01)
        plane_ID = p.loadURDF("plane.urdf")

        path = os.path.abspath(os.path.dirname(__file__))   

        shift = [np.random.randn()*10, np.random.randn()*10, 30 + np.random.random()*20]

        orientation = p.getQuaternionFromEuler(\
                [np.random.randn(1)*1e-1,np.random.randn(1)*1e-1,np.random.rand(1)*1e-1])
        self.bot_id = p.loadURDF(os.path.join(path, "booster.xml"),\
            shift,\
            orientation)
        #self.create_rocket()

        info = p.getJointInfo(self.bot_id, 0)

        print(info)
        p.changeDynamics(self.bot_id,-1, lateralFriction=self.k_friction)

        self.fuel = 100.0
        self.kg_per_kN = 0.3 / 240 # g/(kN*s). 240 is due to the 240 Hz time step in the physics simulatorchangeDynamicsp.change

        # get rid of damping
        num_links = 3
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
            print(thrust)
            # apply thrust to the axis of the rocket's bell nozzle
            p.applyExternalForce(self.bot_id, -1, [0.0, 0.0, thrust],\
                    posObj=[0,0,1.0],flags=p.LINK_FRAME)

            # decrement fuel proportional to the thrust used
            self.fuel -= thrust * self.kg_per_kN

            p.changeDynamics(self.bot_id, 1, mass=self.fuel)
            self.fuel = 0.0 if self.fuel <= 0.0 else self.fuel

        # visual indicator of thrust
        p.changeVisualShape(self.bot_id, -1, rgbaColor=[1e-1*thrust, 0.01, 0.01, 1.0])

    def apply_control_thrust(self, thrust_x, thrust_y):
        
        if self.fuel > 0.0:
            if (thrust_x + thrust_y) / self.kg_per_kN > self.fuel:
                thrust_x = np.sign(thrust_x) * np.min([self.fuel/self.kg_per_kN/2, np.abs(thrust_x)])
                self.fuel -= np.abs(thrust_x) * self.kg_per_kN
                thrust_y = np.sign(thrust_y) * np.min([self.fuel/self.kg_per_kN/2, np.abs(thrust_y)])
                self.fuel -= np.abs(thrust_y) * self.kg_per_kN

            p.applyExternalForce(self.bot_id, 1, [thrust_x, thrust_y, 0.0],\
                    posObj=[0.0, 0.0, 5.5], flags=p.LINK_FRAME)

            p.changeDynamics(self.bot_id, 1, mass=self.fuel)
            self.fuel = 0.0 if self.fuel <= 0.0 else self.fuel

    def step(self, action):
        
        self.apply_thrust(action[2])


        self.apply_control_thrust(action[0], action[1])

        p.stepSimulation()
        
        velocity = p.getBaseVelocity(self.bot_id)

        obs, reward, info = None, None, None

        done = False
        return obs, reward, done, info

    


if __name__ == "__main__":

    env = BoosterBackEnv(render=True)
    env.reset()

    import pdb; pdb.set_trace()
    for cc in range(1000):
        action = [10 * np.random.randn(), 10. *np.random.randn(), 500]
        env.step(action)
        time.sleep(0.01)
    import pdb; pdb.set_trace()
