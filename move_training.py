import numpy as np
import gymnasium as gym
import actions.move_actions as move_actions
import envs.move_env as move_env
import address
import pyMeow as pm
import pydirectinput as pdir
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import random

process = address.process

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
        obs = self.locals['obs_tensor']
        print(60*'=')
        print(f"Step: {self.total_steps}, Current Reward: {reward} | {normalized_reward} ,  Average Reward: {average_reward} | {average_normalized_reward} , Gamma: {gamma}")
        print(f"Player action: {player_action} , Enemy Action: {enemy_action} , Obs: {obs}")
        if self.locals.get("dones", [False])[0]:
            print("Character died. Stopping training.")
            self.end_flag = True
            return False
        return True

class MoveEnvTrain(move_env.MoveEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(MoveEnvTrain, self).__init__(boss_name="iudex", speed = 1)
        self.loc = [141.87632751464844,-68.88593292236328,593.8687133789062]
        self.episode_length = 100
        self.hp_randomizer = 100
        self.hp_lock = pm.r_int(process,address.Address.max_hp)

    
    def random_point_2d(self,x0, z0, max_distance):
        # Generate a random distance
        distance = random.uniform(0, max_distance)

        # Generate a random angle
        angle = random.uniform(0, 2 * np.pi)

        # Convert polar coordinates to Cartesian coordinates
        x = x0 + distance * np.cos(angle)
        z = z0 + distance * np.sin(angle)

        return (x,z)
    
    def random_hp_and_estus_stamina(self):
        return random.randint(0,1) , random.randint(150,self.maximum_hp) , random.randint(0,self.maximum_stamina)
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        action_name = address.read_string(process,address.Address.player_animation_name)
        if pm.r_int(process , address.Address.hp)<=0:
            done = True
            self.release_key()
        # if self.hp_randomizer == 0:
        #     estus_count , self.hp_lock , max_stamina = self.random_hp_and_estus_stamina()
        #     pm.w_int(process , address.Address.estus , estus_count)
        #     pm.w_int(process , address.Address.max_stamina_permanent, max_stamina)
        #     self.hp_randomizer = 100
        # pm.w_int(process , address.Address.hp , self.hp_lock)           
        if self.episode_length == 0:
            x ,z = self.random_point_2d(self.loc[0],self.loc[2],15)
            pm.w_float(process , address.Address.X , x)
            pm.w_float(process , address.Address.Z , z)
            pm.w_float(process , address.Address.Y , self.loc[1])
            self.episode_length = 100
        self.episode_length -= 1
        # self.hp_randomizer -= 1
        if self.prev_action != action:
            self.release_key()
            for act in move_actions.sub_actions[action]:
                pdir.keyDown(act)

        # hp = self.state[0]
        # stamina = self.state[1]
        # estus = self.state[2]
        time.sleep(0.1)
        distance = self.state[0]
        # attacking = self.state[4]

        reward += (5-distance)*5

        if distance > 3 and action == 0:
            reward += 15
        if distance > 3 and action == 1:
            reward -= 15
        if distance < 3 and (action == 2 or action == 3):
            reward += 10
        if action == 4:
            reward -= 10
        # # if stamina == 0:
        # #     if action <= 15 and action >= 8:
        # #         reward -= 15
        # if hp == 0 and estus == 1:
        #     reward -= abs(distance-30) - 7
        #     if attacking == 1 and action == 1:
        #         reward += 20
        #     # if attacking == 0 and action == 9:
        #     #     reward += 20
        # else:
        #     reward += (4 - distance)*7
        #     if distance < 5:
        #         if action ==2 or action==3:
        #             reward += 20
        #         if action <= 1:
        #             reward += 4
        #         if action == 4:
        #             reward -= 8
        #     if attacking == 1:
        #         if action >= 2 and action <= 3:
        #             reward += 20
        #     else:
        #         if distance > 5:
        #             if action == 0:
        #                 reward += 25
        #         # if distance > 10:
        #             # if action == 8:
        #             #     reward += 30
        # # if attacking == 1:
        #     # if distance < 3 and action <=13 and action >=10:
        #     #     reward -= 20
        #     # if action == 8:
        #     #     reward -= 10

        if 'fall' in action_name.lower():
            reward -= 50
        truncated = False
        self.prev_action = action
        print(f'action: {move_actions.sub_actions[action]} , distance: {distance}')
        self.reset()
        return self.read_from_memory(), reward, done, truncated , info

env = MoveEnvTrain(speed=2)

env = DummyVecEnv([lambda: MoveEnvTrain()])

# normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.95)

normalized_env = VecNormalize.load("vec_normalize_stats_move_v6.pkl", env)

custom_callback = RewardCallback(env= normalized_env,verbose=1)

ent_coef = 0.01

model = PPO.load("ppo_dark_souls_move_v6", env=normalized_env)

params = model.get_parameters()

model = PPO('MlpPolicy',learning_rate=0.0005,n_steps=512 , batch_size=64 ,  n_epochs=20 , ent_coef=ent_coef ,env = normalized_env, verbose=1)

model.set_parameters(params)

model.learn(total_timesteps=4096 , callback=custom_callback)

pdir.keyUp('w')
pdir.keyUp('s')
pdir.keyUp('d')
pdir.keyUp('a')
pdir.keyUp('space')

normalized_env.save("vec_normalize_stats_move_v6.pkl")

model.save("ppo_dark_souls_move_v6")