import numpy as np
import gymnasium as gym
import actions.act_actions as act_actions
import envs.act_env as act_env
import address
import pyMeow as pm
import pydirectinput as pdir
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import random
from envs.act_env import QLearningAgent

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
        print(f"Step: {self.total_steps}, Current Reward: {reward} | {normalized_reward} ,  Average Reward: {average_reward} | {average_normalized_reward} , Gamma: {gamma}")
        print(f"Player action: {player_action} , Enemy Action: {enemy_action} , Obs: {obs}")
        print(60*'=')
        if self.locals.get("dones", [False])[0]:
            print("Character died. Stopping training.")
            self.end_flag = True
            return False
        return True


class ActEnvTrain(act_env.ActEnv):
    def __init__(self , boss_name="iudex", speed = 1):
        super(ActEnvTrain, self).__init__('Q_table_dark_souls_dodge_v1_phase2.npy')
        self.speed = speed
        self.hp_lock = pm.r_int(process,address.Address.max_hp)
        self.estus_regeneration = 100
    
    def step(self , action):
        reward = 0
        done = False
        info = {}
        if 'death' in address.read_string(process , address.Address.player_animation_name).lower():
            done = True
        if pm.r_int(process , address.Address.iudex_hp)<=0 or pm.r_int(process , address.Address.hp)<=0:
            done = True
        dodging = self.dodge_model.choose_action(self.dodge_env.state)
        attacking = self.state[0]
        dodged = self.state[1]
        estus = self.state[2]
        stamina_low = self.state[3]
        hp_low = self.state[4]
        prev_hp = pm.r_int(process , address.Address.hp)
        if attacking == 1 and dodged == 0:
            self.dodge_env.step(dodging)
        else:
            if action == 0:
                pdir.press(act_actions.Actions[action][0])
                time.sleep(0.3)
            elif action == 1:
                if estus == 0:
                    time.sleep(1)
                else:
                    pdir.press(act_actions.Actions[action][0])
                    time.sleep(0.3)
            else:
                time.sleep(0.1)
        hp = pm.r_int(process , address.Address.hp)
        if attacking == 1 and dodged == 0:
            if action == 0:
                reward += 5
            else:
                reward -= 5
        else:
            if hp_low == 1:
                if estus == 1:
                    if action == 1:
                        reward += 5
                    if action == 0:
                        reward -= 5
                else:
                    if action == 1:
                        reward -= 50
                    else:
                        if action == 0:
                            reward += 8
                        if action == 2:
                            reward += 5
            if stamina_low == 1:
                if action == 0:
                    reward -= 40
                else:
                    reward += 5
            if hp_low == 0 and stamina_low == 0:
                if action == 0:
                    reward += 8
                else:
                    if action == 1:
                        reward -= 50
                    else:
                        reward += 5
            if hp_low == 0 and action == 1:
                reward -= 80
        if prev_hp > hp and action == 0:
            reward -= 40
        if prev_hp > hp and action == 1:
            reward -= 40
        if hp < 400:
            pm.w_int(process , address.Address.hp , 454)
        if estus == 0:
            # if self.estus_regeneration == 0:
            pm.w_int(process, address.Address.estus, 4)
                # self.estus_regeneration = 40
            # self.estus_regeneration -= 1
        if dodging == 0:
            print(f'action: [space]')
        else:
            print(f'action: {act_actions.Actions[action]}')
        truncated = False
        if pm.r_int(address.process, address.Address.iudex_hp) < 680:
            pm.w_int(address.process, address.Address.iudex_hp , pm.r_int(address.process, address.Address.iudex_max_hp))
        self.reset()
        return self.read_from_memory(), reward, done, truncated , info
    
env = ActEnvTrain('',speed=1)

# env = DummyVecEnv([lambda: ActEnvTrain()])

# # normalized_env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.95)

# normalized_env = VecNormalize.load("vec_normalize_stats_act_v2.pkl", env)

# custom_callback = RewardCallback(env= normalized_env,verbose=1)

# ent_coef = 0.02

# model = PPO.load("ppo_dark_souls_act_v2", env=normalized_env)

# params = model.get_parameters()

# model = PPO('MlpPolicy',learning_rate=0.0005,n_steps=1024 , batch_size=128 ,  n_epochs=20 , ent_coef=ent_coef ,env = normalized_env, verbose=1)

# model.set_parameters(params)

# model.learn(total_timesteps=8192 , callback=custom_callback)

# normalized_env.save("vec_normalize_stats_act_v2.pkl")

# model.save("ppo_dark_souls_act_v2")

agent = QLearningAgent(env.observation_space.shape, env.action_space,epsilon=1,gamma=0.8)
agent.q_table = np.load('Q_table_dark_souls_act_v1_phase2.npy')
# Training loop
episodes = 40
steps = 128
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    for _ in range(steps):
        action = agent.choose_action(state)
        # action = get_action()
        next_state, reward, done, truncated, _ = env.step(action)
        agent.learn(state, action, reward, next_state,  done)
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

env.close()

np.save('Q_table_dark_souls_act_v1_phase2', agent.q_table)