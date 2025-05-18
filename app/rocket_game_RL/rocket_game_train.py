from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import FlattenObservation
import pygame
from env.rocketRL_env import RocketRLEnv
def train_and_test():
    # Training
    print("Training agent...")
    train_env = FlattenObservation(RocketRLEnv(render_mode=None, grid_height=6, grid_width=10))
    train_env = make_vec_env(lambda: train_env, n_envs=1)

    model = DQN("MlpPolicy", train_env, verbose=1, learning_rate=1e-3, buffer_size=10000)
    model.learn(total_timesteps=50000)
    model.save("rocket_dqn_model")

    # Evaluation
    print("Testing agent...")
    test_env = FlattenObservation(RocketRLEnv(render_mode="human", grid_height=6, grid_width=10))
    obs, _ = test_env.reset()
    model = DQN.load("rocket_dqn_model")

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        if terminated:
            obs, _ = test_env.reset()


    test_env.close()


if __name__ == "__main__":
    train_and_test()
