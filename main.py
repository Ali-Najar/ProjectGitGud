import player_xyz as pxyz

pxyz.teleport_to_boss()

import gymnasium as gym
import numpy as np
import actions.actions as actions
import address
import pyMeow as pm
import pydirectinput as pdir
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize

process = address.process
keys = ['w','s','a','d','space','j','k','o','p']

class RewardCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.normalized_rewards = []
        self.total_steps = 0
        self.env = env
        self.end_flag = False
    def _on_step(self) -> bool:
        # Extract the reward from the environment
        reward = self.env.get_original_reward()
        normalized_reward = self.locals['rewards']
        self.rewards.append(reward)
        self.normalized_rewards.append(normalized_reward)
        self.total_steps += 1

        # Calculate the average reward
        average_reward = sum(self.rewards) / len(self.rewards)
        average_normalized_reward = sum(self.normalized_rewards) / len(self.normalized_rewards)

        enemy_action = address.read_string(process,address.Address.iudex_animation_name)
        player_action = address.read_string(process,address.Address.player_animation_name)
        gamma = self.env.gamma
        # Log the current step and average reward
        print(f"Step: {self.total_steps}, Current Reward: {reward} | {normalized_reward} ,  Average Reward: {average_reward} | {average_normalized_reward} , Gamma: {gamma} , Player action: {player_action} , Enemy Action: {enemy_action}\n")
        if self.locals.get("dones", [False])[0]:
            print("Character died. Stopping training.")
            self.end_flag = True
            return False
        return True

class GameEnv(gym.Env):
    def __init__(self , boss_name="iudex", speed = 1):
        super(GameEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,shape=(23,))
        self.action_space = gym.spaces.Discrete(len(actions.Actions))
        self.speed = speed
        
    def reset(self , seed=None , options=None):
        # Initialize the game state
        super().reset(seed=seed)
        pm.w_int(process , address.Address.hp , pm.r_int(process , address.Address.max_hp))
        # pm.w_int(process , address.Address.iudex_hp , pm.r_int(process , address.Address.iudex_max_hp))
        pm.w_int(process , address.Address.fp , pm.r_int(process , address.Address.max_fp))
        pm.w_int(process , address.Address.stamina , pm.r_int(process , address.Address.max_stamina))
        pm.w_int(process , address.Address.estus , 3)
        # pm.w_float(process , address.Address.player_speed , self.speed)
        # pm.w_float(process , address.Address.iudex_speed , self.speed)
        
        self.state = self.read_from_memory()

        return self.state , {}
    def read_from_memory(self):
        return np.array([
                        pm.r_int(process , address.Address.hp),
                        pm.r_int(process , address.Address.iudex_hp),
                        pm.r_int(process , address.Address.fp),
                        pm.r_int(process , address.Address.stamina),
                        pm.r_int(process , address.Address.estus),
                        pm.r_float(process , address.Address.O),
                        pm.r_float(process , address.Address.X),
                        pm.r_float(process , address.Address.Y),
                        pm.r_float(process , address.Address.Z),
                        pm.r_float(process , address.Address.CamX),
                        pm.r_float(process , address.Address.CamY),
                        pm.r_float(process , address.Address.CamZ),
                        pm.r_float(process , address.Address.iudex_O),
                        pm.r_float(process , address.Address.iudex_X),
                        pm.r_float(process , address.Address.iudex_Y),
                        pm.r_float(process , address.Address.iudex_Z),
                        pm.r_int(process , address.Address.player_animation),
                        pm.r_float(process , address.Address.player_animation_time),
                        pm.r_float(process , address.Address.player_max_animation_time),
                        pm.r_int(process , address.Address.iudex_animation),
                        pm.r_float(process , address.Address.iudex_animation_time),
                        pm.r_float(process , address.Address.iudex_max_animation_time),
                        pm.r_int(process , address.Address.LockOn)
                        ])  
    def step(self , action):
        reward = 0
        done = False
        info = {}
        prev_hp = self.state[0]
        prev_iudex_hp = self.state[1]
        prev_estus = self.state[4]
        prev_lockOn = self.state[22]
        enemy_action = address.read_string(process,address.Address.iudex_animation_name)
        print(actions.Actions[action])
        self.release_key()
        if action < 8 or (action >12 and action < 17):
            pdir.keyDown(actions.Actions[action][0])
            if len(actions.Actions[action])>1:
                pdir.keyDown(actions.Actions[action][1])
        if action >= 8 and action <= 11:
            pdir.keyDown(actions.Actions[action][0])
            pdir.press(actions.Actions[action][1])
            pdir.keyUp(actions.Actions[action][0])
        if action == 12 or action == 17 or action == 18 or action == 20 or action == 21:
            pdir.press(actions.Actions[action][0])
        if action == 19:
            pdir.keyDown(actions.Actions[action][0])
        if (action >= 8 and action <=12) or (action >= 18 and action<= 20):
            delay = max(pm.r_float(process , address.Address.player_max_animation_time) - 0.5 , 0)
            time.sleep(delay/self.speed)
            pdir.keyUp('h')
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        if pm.r_int(process , address.Address.iudex_hp)==0:
            reward += 1000
        if pm.r_int(process , address.Address.hp)==0:
            reward -= 1000
        if pm.r_int(process , address.Address.hp) < prev_hp:
            reward += 3*(pm.r_int(process , address.Address.hp) - prev_hp)
        else:
            reward += pm.r_int(process , address.Address.hp) - prev_hp
        reward += 10*(prev_iudex_hp - pm.r_int(process , address.Address.iudex_hp))
        if pm.r_int(process , address.Address.stamina) < 10:
            reward -= 10
        if pm.r_int(process , address.Address.estus) < prev_estus:
            reward -= 1
        if prev_estus == 0 and action == 20:
            reward -= 10
        if pm.r_int(process , address.Address.LockOn) != prev_lockOn:
            if prev_lockOn == 0:
                reward += 500
            else:
                reward -= 200
        if pm.r_int(process , address.Address.LockOn) == 0:
            reward -= 200
        if pm.r_int(process , address.Address.iudex_hp)<700 or pm.r_int(process , address.Address.hp)<250:
            if pm.r_int(process , address.Address.iudex_hp)<700:
                reward += 10000
                pm.w_int(process , address.Address.iudex_hp , pm.r_int(process , address.Address.iudex_max_hp))           
            # if pm.r_int(process , address.Address.hp)<250:
            #     reward -= 50
            self.reset()
        if 'attack' in enemy_action.lower() and action>=8 and action<=12:
            reward += 500
        if 'attack' in enemy_action.lower() and not (action>=8 and action<=12):
            reward -= 300
        if action ==18 or action ==19:
            if prev_iudex_hp - pm.r_int(process , address.Address.iudex_hp) < 0:
                if pm.r_int(process , address.Address.LockOn) == 0:
                    reward += 200
                else:
                    reward += 500
            else:
                if pm.r_int(process , address.Address.LockOn) == 0:
                    reward -= 500
        truncated = False
        if 'fall' in address.read_string(process,address.Address.player_animation_name).lower():
            reward -= 1000
        return self.read_from_memory(), reward, done, truncated , info
    def release_key(self):
        for key in keys:
            pdir.keyUp(key)

env = GameEnv(speed=1)

env = DummyVecEnv([lambda: GameEnv()])

normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.95)

custom_callback = RewardCallback(env= normalized_env,verbose=1)

n_iters = 20
ent_coef = 0.01
# for i in range(n_iters):
#     print(f"ITERATION: {i}")
#     if custom_callback.end_flag:
#         time.sleep(15)
#         print("KIR")
#         pxyz.teleport_to_boss()
model = PPO.load("ppo_dark_souls", env=normalized_env)

params = model.get_parameters()

model = PPO('MlpPolicy',learning_rate=0.005,n_steps=128 , batch_size=64 ,  n_epochs=20 , ent_coef=ent_coef ,env = normalized_env, verbose=1)

model.set_parameters(params)


model.learn(total_timesteps=1024 , callback=custom_callback)

model.save("ppo_dark_souls")

# ent_coef *= 0.9