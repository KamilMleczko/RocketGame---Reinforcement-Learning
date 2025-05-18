from stable_baselines3 import DQN
from env.rocketRL_env import RocketRLEnv
env = RocketRLEnv(render_mode="human", grid_height=10, grid_width=15)
model = DQN.load("rocket_dqn_model")  

obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
