import gymnasium as gym
import actions.cam_actions as cam_actions
import numpy as np
import address
import pyMeow as pm
import random
import pydirectinput as pdir
import time

process = address.process

class LockEnv(gym.Env):
    def __init__(self , boss_name="iudex", speed = 1):
        super(LockEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(2,))
        self.action_space = gym.spaces.Discrete(len(cam_actions.Lock_On))
        self.speed = speed
        
    def reset(self , seed=None , options=None):
        # Initialize the game state
        super().reset(seed=seed)
        self.state = self.read_from_memory()

        return self.state , {}
    
    def read_from_memory(self):
        angle = self.angle_calc()
        return np.array([ angle,
                        pm.r_int16(process , address.Address.LockOn)
                        ])  
    
    def angle_calc(self):
        x = pm.r_float(process , address.Address.iudex_X) - pm.r_float(process , address.Address.X)
        y = pm.r_float(process , address.Address.iudex_Y) - pm.r_float(process , address.Address.Y)
        z = pm.r_float(process , address.Address.iudex_Z) - pm.r_float(process , address.Address.Z)
        cx = pm.r_float(process , address.Address.CamX)
        cy = pm.r_float(process , address.Address.CamY)
        cz = pm.r_float(process , address.Address.CamZ)

        cam = np.array([cx , cy , cz])
        second_vec = np.array([x , y , z])

        dot_prod = np.dot(cam , second_vec)
        cam_norm = np.linalg.norm(cam)
        second_vec_norm = np.linalg.norm(second_vec)
        angle = np.arccos(dot_prod/(cam_norm*second_vec_norm))

        return angle
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        if action == 0:
            pdir.press(cam_actions.Lock_On[action][0])
        self.reset()
        truncated = False
        time.sleep(0.75)
        return self.read_from_memory(), reward, done, truncated , info
    