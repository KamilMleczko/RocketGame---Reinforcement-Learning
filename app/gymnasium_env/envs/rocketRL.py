import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from enum import Enum


class Actions(Enum):
    up = 0
    down = 1
    stay = 2


class RocketRLEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_height=5, grid_width=10, difficulty_increase_rate=0.1):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.window_size = (700, 500)
        self.rocket_pos = np.array([1, grid_height // 2])
        self.rocks = []
        self.initial_speed = 1
        self.current_speed = self.initial_speed
        self.difficulty_increase_rate = difficulty_increase_rate
        self.steps = 0
        self.score = 0
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.observation_space = spaces.Dict({
            "rocket": spaces.Box(low=0, high=np.array([grid_width-1, grid_height-1]), shape=(2,), dtype=int),
            "rocks": spaces.Box(
                low=np.zeros((grid_width, 2), dtype=int),
                high=np.ones((grid_width, 2), dtype=int) * np.array([grid_width-1, grid_height-1]),
                shape=(grid_width, 2),
                dtype=int
            ),
            "speed": spaces.Box(low=0, high=10.0, shape=(1,), dtype=float)
        })

        self.action_space = spaces.Discrete(3)
        self._action_to_direction = {
            Actions.up.value: np.array([0, -1]),
            Actions.down.value: np.array([0, 1]),
            Actions.stay.value: np.array([0, 0]),
        }

        self.np_random = np.random.default_rng()

    def _get_obs(self):
        rocks_array = np.full((self.grid_width, 2), -1, dtype=int)
        for i, rock in enumerate(self.rocks[:self.grid_width]):
            rocks_array[i] = rock
        return {
            "rocket": self.rocket_pos,
            "rocks": rocks_array,
            "speed": np.array([self.current_speed], dtype=float)
        }

    def _get_info(self):
        return {
            "rocks_count": len(self.rocks),
            "speed": self.current_speed,
            "steps": self.steps,
            "score": self.score
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rocket_pos = np.array([1, self.grid_height // 2])
        self.rocks = []
        self._generate_new_rocks()
        self.current_speed = self.initial_speed
        self.steps = 0
        self.score = 0
        return self._get_obs(), self._get_info()

    def _generate_new_rocks(self):
        num_new_rocks = self.np_random.integers(1, 4)
        for _ in range(num_new_rocks):
            rock_y = self.np_random.integers(0, self.grid_height)
            self.rocks.append(np.array([self.grid_width - 1, rock_y]))

    def step(self, action):
        self.steps += 1
        self.score += 1

        direction = self._action_to_direction[int(action)]

        self.rocket_pos = np.clip(self.rocket_pos + direction, [1, 0], [1, self.grid_height - 1])

        self.rocks = [rock - np.array([self.current_speed, 0]) for rock in self.rocks]
        self.rocks = [rock for rock in self.rocks if rock[0] >= 0]

        if self.np_random.random() < 0.3:
            self._generate_new_rocks()

        terminated = any(np.array_equal(self.rocket_pos, np.floor(rock).astype(int)) for rock in self.rocks)

        if self.steps % 20 == 0:
            self.current_speed += self.difficulty_increase_rate

        reward = -1.0 if terminated else 0.1
        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 0, 0))
        pix_x = self.window_size[0] / self.grid_width
        pix_y = self.window_size[1] / self.grid_height

        rocket_cx = (self.rocket_pos[0] + 0.5) * pix_x
        rocket_cy = (self.rocket_pos[1] + 0.5) * pix_y
        tw, th = pix_x * 0.8, pix_y * 0.8

        points = [
            (rocket_cx + tw / 2, rocket_cy),
            (rocket_cx - tw / 2, rocket_cy - th / 2),
            (rocket_cx - tw / 2, rocket_cy + th / 2),
        ]
        pygame.draw.polygon(canvas, (0, 100, 255), points)

        for rock in self.rocks:
            rock_size_x = pix_x * 0.8
            rock_size_y = pix_y * 0.8
            cx = (rock[0] + 0.5) * pix_x
            cy = (rock[1] + 0.5) * pix_y
            pygame.draw.rect(
                canvas,
                (255, 50, 50),
                pygame.Rect(cx - rock_size_x / 2, cy - rock_size_y / 2, rock_size_x, rock_size_y)
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

