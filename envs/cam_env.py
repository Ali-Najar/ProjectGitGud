import gymnasium as gym
import actions.cam_actions as cam_actions
import numpy as np
import address
import pyMeow as pm
import random
import pydirectinput as pdir
import time
pdir.PAUSE = 0.0
process = address.process

class CamEnv(gym.Env):
    def __init__(self , boss_name="iudex", speed = 1):
        super(CamEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(7,))
        self.action_space = gym.spaces.Discrete(len(cam_actions.Camera))
        self.speed = speed
        self.keys = ['j','k','o','p']
        
    def reset(self , seed=None , options=None):
        # Initialize the game state
        super().reset(seed=seed)
        self.state = self.read_from_memory()

        return self.state , {}
    
    def read_from_memory(self):
        angle = self.angle_calc()
        x = pm.r_float(process , address.Address.iudex_X) - pm.r_float(process , address.Address.X)
        y = pm.r_float(process , address.Address.iudex_Y) - pm.r_float(process , address.Address.Y)
        z = pm.r_float(process , address.Address.iudex_Z) - pm.r_float(process , address.Address.Z)
        nx , ny , nz = self.normalize_vector(x, y, z)
        cx = pm.r_float(process , address.Address.CamX)
        cy = pm.r_float(process , address.Address.CamY)
        cz = pm.r_float(process , address.Address.CamZ)
        return np.array([
                        nx, ny, nz, cx, cy, cz,
                        angle,
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
        angle = np.arccos(np.clip(dot_prod/(cam_norm*second_vec_norm), -1.0, 1.0))

        return angle
    
    def normalize_vector(self ,x, y , z):
        norm = np.sqrt(x * x + y * y + z * z)
        nx = x / norm
        ny = y / norm
        nz = z / norm
        return nx , ny, nz

    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True

        # 1. Execute Action
        if action < 4:
            pdir.keyDown(cam_actions.Camera[action][0])
            
        time.sleep(0.1)
        
        if action < 4:
            self.release_key()

        # 2. Update State (CRITICAL: Do not reset here)
        self.state = self.read_from_memory()

        # 3. Check Termination
        if pm.r_int(process, address.Address.hp)<=0:
            done = True

        truncated = False
        
        return self.state, reward, done, truncated , info
    
    def release_key(self):
        for key in self.keys:
            pdir.keyUp(key)
