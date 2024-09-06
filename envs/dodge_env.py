import gymnasium as gym
import actions.act_actions as act_actions
import numpy as np
import address
import pyMeow as pm
import random
import pydirectinput as pdir
import time

process = address.process

class DodgeEnv(gym.Env):
    def __init__(self , boss_name="iudex", speed = 1):
        super(DodgeEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(3,))
        self.action_space = gym.spaces.Discrete(len(act_actions.Dodge))
        self.speed = speed
        
    def reset(self , seed=None , options=None):
        # Initialize the game state
        super().reset(seed=seed)
        self.state = self.read_from_memory()

        return self.state , {}
    def read_from_memory(self):
        attacking = 0
        if 'attack' in address.read_string(process,address.Address.iudex_animation_name).lower():
            attacking = 1
        if 'atk' in address.read_string(process,address.Address.iudex_animation_name).lower():
            attacking = 1
        time = pm.r_float(process , address.Address.iudex_animation_time)
        max_time = pm.r_float(process , address.Address.iudex_max_animation_time)
        return np.array([
                        attacking,
                        time,
                        max_time
                        ])  
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        if action == 0:
            pdir.press(act_actions.Dodge[action][0])
            time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)-0.2))
        self.reset()
        truncated = False
        return self.read_from_memory(), reward, done, truncated , info
    
    def release_key(self):
        for key in self.keys:
            pdir.keyUp(key)
