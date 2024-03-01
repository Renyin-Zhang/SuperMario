from stable_baselines3 import PPO

# Import necessary wrappers
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)

# Vectorize the environment using DummyVecEnv
env = DummyVecEnv([lambda: env])

# Apply frame stacking
env = VecFrameStack(env, 4, channels_order='last')

# Load the pre-trained model
model = PPO.load("./model/ppo_mario550k", env=env)


RENDER_EVERY = 5

# Test the trained agent
obs = env.reset()
for step in range(10000):
    action, _ = model.predict(obs.copy(), deterministic=True)

    if action not in [1, 5]:
        print(f"Step {step}: Action = {action}")

    obs, reward, done, info = env.step(action)

    mario_world_x = info[0]["x_pos"]
    mario_world_y = info[0]["y_pos"]
    # Also, you can get Mario's status (small, tall, fireball) from info too.
    mario_status = info[0]["status"]
    print("Mario's location in world:",
        mario_world_x, mario_world_y, f"({mario_status} mario)")


    if step % RENDER_EVERY == 0:
        env.render()
    if done:
        print("distance: ", mario_world_x, "/coins: ", info[0]["coins"], "/score: ", info[0]["score"], "/time: ", info[0]["time"])
        break
        obs = env.reset()
env.close()
