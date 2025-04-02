import gymnasium as gym
import actions.act_actions as act_actions
import numpy as np
import address
import pyMeow as pm
import random
import pydirectinput as pdir
import time
import envs.dodge_env as dodge_env
import keyboard

process = address.process

class ActEnv(gym.Env):
    def __init__(self , dodge_table ,attack_max = 3,boss_name="iudex", speed = 1):
        super(ActEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(7,))
        self.action_space = gym.spaces.Discrete(len(act_actions.Actions))
        self.speed = speed
        self.maximum_stamina = pm.r_int(process,address.Address.max_stamina_permanent)
        self.maximum_hp = pm.r_int(process,address.Address.max_hp_permanent)
        self.dodge_env = dodge_env.DodgeEnv()
        self.dodge_model = dodge_env.QLearningAgent(self.dodge_env.observation_space.shape, self.dodge_env.action_space,epsilon=0)
        self.dodge_model.q_table = np.load(dodge_table)
        self.dodge_env.reset()
        self.attack_max = attack_max
        self.attack_counter = 0
        
        
    def reset(self , seed=None , options=None):
        # Initialize the game state
        super().reset(seed=seed)
        self.state = self.read_from_memory()

        return self.state , {}
    def read_from_memory(self):
        self.dodge_env.reset()
        attacking = self.dodge_env.state[0]
        dodged = self.dodge_env.state[1]
        ratio = self.dodge_env.state[2]
        max_time = self.dodge_env.state[3]
        estus = 1
        if pm.r_int(process , address.Address.estus) == 1:
            estus = 0
        stamina = pm.r_int(process , address.Address.stamina)
        stamina_low = 0
        if stamina < 0.4 * self.maximum_stamina:
            stamina_low = 1
        hp = pm.r_int(process , address.Address.hp)
        hp_low = 0
        if hp < 0.5 * self.maximum_hp:
            hp_low = 1
        return np.array([
                        attacking,
                        dodged,
                        estus,
                        stamina_low,
                        hp_low,
                        ratio,
                        max_time
                        ])  
    
    # def distance(self, x, y ,z ,ex ,ey ,ez):
    #     a = np.array([x,y,z])
    #     b = np.array([ex,ey,ez])
    #     return np.linalg.norm(a-b)
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=1 or pm.r_int(process , address.Address.hp)<=1:
            done = True
        dodging = self.dodge_model.choose_action(self.dodge_env.state)
        attacking = self.state[0]
        dodged = self.state[1]
        estus = self.state[2]
        stamina_low = self.state[3]
        hp_low = self.state[4]
        if attacking == 1 and dodged == 0:
            self.dodge_env.step(dodging)
            self.attack_counter = 0
        else:
            if action == 2:
                time.sleep(0.1)
            else:
                if action == 0 and self.attack_counter != self.attack_max:
                    pdir.press(act_actions.Actions[action][0])
                    time.sleep(0.3)
                elif action == 1 and hp_low == 1 and estus == 1:
                    pdir.press(act_actions.Actions[action][0])
                    time.sleep(0.3)
                else:
                    time.sleep(0.1)
            if action == 0 and self.attack_max != self.attack_counter:
                self.attack_counter += 1
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
    def __init__(self, observation_space_shape, action_space, bins=(2, 2, 2, 2, 2, 30, 50), alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.bins = np.array(bins)
        self.q_table = np.zeros(tuple(self.bins) + (action_space.n,))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.observation_space_low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # adjust based on your environment's observation space low
        self.observation_space_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0])  # adjust based on your environment's observation space high

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