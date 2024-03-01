from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from gym import RewardWrapper, Wrapper

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'ppo_mario{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    
CHECKPOINT_DIR = './model/'
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

class FrameSkip(Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super(FrameSkip, self).__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action and sum reward"""
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info



class PositionalRewardWrapper(RewardWrapper):
    def __init__(self, env, death_penalty=-25.0, move_right_bonus=250.0,
                 jump_bonus=1.0, action_penalty=-500.0, repeat_action_penalty=-170.0):
        super(PositionalRewardWrapper, self).__init__(env)
        self.death_penalty = death_penalty
        self.last_x_position = None
        self.last_y_position = None
        self.jump_bonus = jump_bonus
        self.last_action = None
        self.action_penalty = action_penalty
        self.move_right_bonus = move_right_bonus
        self.repeat_action_count = 0
        self.repeat_action_penalty = repeat_action_penalty

    def step(self, action):
        observation, reward, done, info = super().step(action)

        if action in [0, 2, 4, 5, 6]:
            if self.last_action in [0, 2, 4, 5, 6]:
                self.repeat_action_count += 1
            else:
                self.repeat_action_count = 1
        else:
            self.repeat_action_count = 0
        
        if self.repeat_action_count >= 17:
            reward += self.repeat_action_penalty

        if action in [1, 3]:  # Move right
            reward += self.move_right_bonus

        if action in [0, 5, 6]:
            reward += self.action_penalty

        # # Check if Mario has moved upwards (jumped)
        # current_y_position = self.env.unwrapped._y_position
        # if self.last_y_position is not None and current_y_position < self.last_y_position:
        #     reward += self.jump_bonus

        # # Update the last y-coordinate for the next step
        # self.last_y_position = current_y_position

        # If Mario dies and restarts, apply the death penalty
        if done:
            reward += self.death_penalty

        # Record the last action for the next step
        self.last_action = action

        return observation, self.reward(reward), done, info

    def reward(self, reward):
        # Extract Mario's x position from the info dict to check for deaths
        # If Mario's x position resets back to the start, he likely died
        # x_position = self.env.unwrapped._x_position
        # if self.last_x_position is not None and x_position < self.last_x_position:
        #     reward += self.death_penalty

        # self.last_x_position = x_position
        return reward
    
frames_to_skip = 4

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = PositionalRewardWrapper(env)
env = FrameSkip(env, skip=frames_to_skip)

# Vectorize the environment using DummyVecEnv
env = DummyVecEnv([lambda: env])

env = VecFrameStack(env, 4, channels_order='last')

# 1. Adjusting Entropy Coefficient
ent_coef = 0.01

# 2. Adjust Learning Rate
learning_rate = 0.0001

# 3. Adjust Clipping Range
clip_range = 0.2

# 4. Adjust Value Function Coefficient
vf_coef = 0.25

# model = PPO("CnnPolicy", env,
#                 ent_coef=ent_coef, 
#                 learning_rate=learning_rate, 
#                 clip_range=clip_range, 
#                 vf_coef=vf_coef,
#                 n_steps = 512)
# Loading the pre-trained model with the updated hyperparameters
model = PPO.load("./model/ppo_mario650k",
                 env=env, 
                 ent_coef=ent_coef, 
                 learning_rate=learning_rate, 
                 clip_range=clip_range, 
                 vf_coef=vf_coef,
                 n_steps = 512)

model.verbose = 1
model.tensorboard_log = "./tensorboard/"

model.learn(total_timesteps=100000, callback = callback)