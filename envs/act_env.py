import gymnasium as gym
import actions.act_actions as act_actions
import numpy as np
import address
import pyMeow as pm
import random
import pydirectinput as pdir
import time

process = address.process

class ActEnv(gym.Env):
    def __init__(self , dodge_env, boss_name="iudex", speed = 1):
        super(ActEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(8,))
        self.action_space = gym.spaces.Discrete(len(act_actions.Actions))
        self.dodge_env = dodge_env
        self.speed = speed
        self.maximum_stamina = pm.r_int(process,address.Address.max_stamina_permanent)
        self.maximum_hp = pm.r_int(process,address.Address.max_hp_permanent)
        
        
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
        dist = self.distance(x, y, z, ex, ey ,ez)
        dist_far = 1
        if dist < 4:
            dist_far = 0
        attacking = 0
        if 'attack' in address.read_string(process,address.Address.iudex_animation_name).lower():
            attacking = 1
        if 'atk' in address.read_string(process,address.Address.iudex_animation_name).lower():
            attacking = 1
        estus = 1
        if pm.r_int(process , address.Address.estus) == 1:
            estus = 0
        stamina = pm.r_int(process , address.Address.stamina)
        stamina_low = 0
        if stamina < 0.3 * self.maximum_stamina:
            stamina_low = 1
        hp = pm.r_int(process , address.Address.hp)
        iudex_hp = pm.r_int(process , address.Address.iudex_hp)
        return np.array([
                        dist_far,
                        attacking,
                        estus,
                        stamina_low,
                        hp,
                        iudex_hp
                        ])  
    
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
        if action > 0 and action < 4:
            pdir.keyDown(act_actions.Actions[action][0])
            time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)))
            pdir.keyUp(act_actions.Actions[action][0])
        if action == 0:
            pdir.press(act_actions.Actions[action][0])
            time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)))
        self.reset()
        truncated = False
        return self.read_from_memory(), reward, done, truncated , info
    
    def release_key(self):
        for key in self.keys:
            pdir.keyUp(key)
