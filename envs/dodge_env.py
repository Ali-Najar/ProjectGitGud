import gymnasium as gym
import actions.act_actions as act_actions
import numpy as np
import address
import pyMeow as pm
import random
import pydirectinput as pdir
import time
import keyboard

process = address.process

class DodgeEnv(gym.Env):
    def __init__(self , boss_name="iudex", speed = 1):
        super(DodgeEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(4,))
        self.action_space = gym.spaces.Discrete(len(act_actions.Dodge))
        self.speed = speed
        self.action_dodged = False
        self.prev_action = None

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
        time = 0
        max_time = 0
        if abs(pm.r_float(process,address.Address.iudex_max_animation_time) - 5) < 0.1:
            if abs(pm.r_float(process,address.Address.iudex_max_animation_time+0x10) - 5) < 0.1:
                time = pm.r_float(process , address.Address.iudex_animation_time+0x20)
                max_time = pm.r_float(process , address.Address.iudex_max_animation_time+0x20)
            else:
                time = pm.r_float(process , address.Address.iudex_animation_time+0x10)
                max_time = pm.r_float(process , address.Address.iudex_max_animation_time+0x10)
        else:
            time = pm.r_float(process , address.Address.iudex_animation_time)
            max_time = pm.r_float(process , address.Address.iudex_max_animation_time)
        ratio = time/max_time
        dodged = 0
        if self.prev_action is None:
            self.prev_action = address.read_string(process,address.Address.iudex_animation_name)
            self.action_dodged = False
        elif self.prev_action != address.read_string(process,address.Address.iudex_animation_name):
            self.prev_action = address.read_string(process,address.Address.iudex_animation_name)
            self.action_dodged = False
        if self.action_dodged == True:
            dodged = 1
        return np.array([
                        attacking,
                        dodged,
                        ratio,
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
            pdir.keyDown('w')
            pdir.press(act_actions.Dodge[action][0])
            pdir.keyUp('w')
            time.sleep(max(0,pm.r_float(process,address.Address.player_max_animation_time)-1.4))
        else:
            time.sleep(0.2)
        if action == 0:
            self.action_dodged = True
        self.reset()
        truncated = False
        return self.read_from_memory(), reward, done, truncated , info
    
    def release_key(self):
        for key in self.keys:
            pdir.keyUp(key)

def discretize_state(state, bins, observation_space_low, observation_space_high):
    ratios = (state - observation_space_low) / (observation_space_high - observation_space_low)
    new_obs = (ratios * bins).astype(int)
    return tuple(np.clip(new_obs, 0, bins - 1))

def get_action():
    if keyboard.is_pressed('1'):
        pdir.keyUp('1')
        return 0
    return 1

class QLearningAgent:
    def __init__(self, observation_space_shape, action_space, bins=(2, 2, 30, 50), alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.bins = np.array(bins)
        self.q_table = np.zeros(tuple(self.bins) + (action_space.n,))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.observation_space_low = np.array([0.0, 0.0, 0.0, 0.0])  # adjust based on your environment's observation space low
        self.observation_space_high = np.array([1.0, 1.0, 1.0, 10.0])  # adjust based on your environment's observation space high

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()  # Explore
        state_discrete = discretize_state(state, self.bins, self.observation_space_low, self.observation_space_high)
        return np.argmax(self.q_table[state_discrete])  # Exploit

    def learn(self, state, action, reward, next_state, done):
        
        state_discrete = discretize_state(state, self.bins, self.observation_space_low, self.observation_space_high)
        next_state_discrete = discretize_state(next_state, self.bins, self.observation_space_low, self.observation_space_high)

        best_next_action = np.argmax(self.q_table[next_state_discrete])

        target = reward + (self.gamma * self.q_table[next_state_discrete][best_next_action] * (not done))
        self.q_table[state_discrete][action] += self.alpha * (target - self.q_table[state_discrete][action])

        # if done:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        print(f'state: {state} , q-value: {self.q_table[state_discrete]} , reward: {reward} , epsilon: {self.epsilon}')
        print(50*'=')