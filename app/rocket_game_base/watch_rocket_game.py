from gymnasium.envs.registration import register
import gymnasium as gym
from env.rocket_env import RocketEnv
import pygame
import numpy as np

def main():
    #env = RocketEnv(render_mode="human", )
    env = RocketEnv(render_mode="human", grid_height=10, grid_width=15) 
    observation, info = env.reset()

    for _ in range(1000):
        # Take a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Process events to check for window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()
        
        if terminated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()