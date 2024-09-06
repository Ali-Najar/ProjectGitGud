import gymnasium as gym
import actions.move_actions as move_actions
import numpy as np
import address
import pyMeow as pm
import random
import pydirectinput as pdir
import time

process = address.process

class MoveEnv(gym.Env):
    def __init__(self , boss_name="iudex", speed = 1):
        super(MoveEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(1,))
        self.action_space = gym.spaces.Discrete(len(move_actions.sub_actions))
        self.speed = speed
        self.maximum_stamina = pm.r_int(process,address.Address.max_stamina_permanent)
        self.maximum_hp = pm.r_int(process,address.Address.max_hp_permanent)
        self.keys = ['w','s','a','d','space']
        self.prev_action = 4

        
    def reset(self , seed=None , options=None):
        # Initialize the game state
        super().reset(seed=seed)
        self.state = self.read_from_memory()

        return self.state , {}
    
    def read_from_memory(self):
        x = pm.r_float(process , address.Address.X)
        y = pm.r_float(process , address.Address.Y)
        z = pm.r_float(process , address.Address.Z)
        ex = pm.r_float(process , address.Address.iudex_X)
        ey = pm.r_float(process , address.Address.iudex_Y)
        ez = pm.r_float(process , address.Address.iudex_Z)
        # attacking = 0
        # if 'attack' in address.read_string(process,address.Address.iudex_animation_name).lower():
        #     attacking = 1
        # estus = 1
        # if pm.r_int(process , address.Address.estus) == 0:
        #     estus = 0
        # stamina = 1
        # if self.maximum_stamina*0.2 > pm.r_int(process , address.Address.stamina):
        #     stamina = 0
        # hp = 1
        # if self.maximum_hp*0.5 > pm.r_int(process , address.Address.hp):
        #     hp = 0
        dist = self.distance(x, y, z, ex, ey ,ez)

        return np.array([dist])
        # return np.array([
        #                 hp,
        #                 stamina,
        #                 estus,
        #                 dist,
        #                 attacking
        #                 ]) 
     
    def distance(self, x, y ,z ,ex ,ey ,ez):
        a = np.array([x,y,z])
        b = np.array([ex,ey,ez])
        return np.linalg.norm(a-b)
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        if self.prev_action != action:
            self.release_key()
            for act in move_actions.sub_actions[action]:
                pdir.keyDown(act)
        truncated = False
        self.prev_action = action
        time.sleep(0.1)
        return self.read_from_memory(), reward, done, truncated , info
    def release_key(self):
        for key in self.keys:
            pdir.keyUp(key)