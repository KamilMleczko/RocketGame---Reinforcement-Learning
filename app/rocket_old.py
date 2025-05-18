from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


class Actions(Enum):
    up = 0
    down = 1
    stay = 2


class RocketEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_height=5, grid_width=10, difficulty_increase_rate=0.1):
        self.grid_height = grid_height  # The height of the grid
        self.grid_width = grid_width    # The width of the grid
        self.window_size = (700, 500)   # The size of the PyGame window
        self.rocket_pos = np.array([1, grid_height // 2])  # Fixed x position, variable y position
        self.rocks = []                 # List to store positions of rocks
        self.initial_speed = 1          # Initial speed of rocks approaching
        self.current_speed = self.initial_speed
        self.difficulty_increase_rate = difficulty_increase_rate  # Rate at which speed increases
        self.steps = 0                  # Count steps for increasing difficulty
        
        # Observations are dictionaries with the rocket's location and the rocks' locations
        self.observation_space = spaces.Dict(
            {
                "rocket": spaces.Box(low=0, high=np.array([grid_width-1, grid_height-1]), shape=(2,), dtype=int),
                # For rocks, create a Box with proper shape for grid_width number of rocks, each with x,y coordinates
                "rocks": spaces.Box(
                    low=np.zeros((grid_width, 2), dtype=int),
                    high=np.ones((grid_width, 2), dtype=int) * np.array([grid_width-1, grid_height-1]),
                    shape=(grid_width, 2), 
                    dtype=int
                ),
                "speed": spaces.Box(low=0, high=10.0, shape=(1,), dtype=float)
            }
        )

        # We have 3 actions, corresponding to "up", "down", and "stay"
        self.action_space = spaces.Discrete(3)

        # This dictionary maps abstract actions to the direction we will move in
        self._action_to_direction = {
            Actions.up.value: np.array([0, -1]),
            Actions.down.value: np.array([0, 1]),
            Actions.stay.value: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # For rendering
        self.window = None
        self.clock = None

    def _get_obs(self):
        # Create a grid_width x 2 array filled with -1 for rocks
        rocks_array = np.full((self.grid_width, 2), -1, dtype=int)
        
        # Fill in actual rock positions if any
        for i, rock in enumerate(self.rocks):
            if i < self.grid_width:
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
            "steps": self.steps
        }

    def reset(self, seed=None, options=None):
        # Reset the environment state
        super().reset(seed=seed)
        
        # Reset rocket to middle position
        self.rocket_pos = np.array([1, self.grid_height // 2])
        
        # Clear existing rocks
        self.rocks = []
        
        # Initialize rocks at random positions on the right side
        self._generate_new_rocks()
        
        # Reset speed and steps counter
        self.current_speed = self.initial_speed
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _generate_new_rocks(self):
        # Add 1-3 new rocks at the right edge of the grid
        num_new_rocks = self.np_random.integers(1, 4)
        for _ in range(num_new_rocks):
            rock_y = self.np_random.integers(0, self.grid_height)
            self.rocks.append(np.array([self.grid_width - 1, rock_y]))

    def step(self, action):
        self.steps += 1
        
        # Move the rocket according to action
        direction = self._action_to_direction[action]
        # Ensure rocket stays within grid boundaries
        self.rocket_pos = np.clip(
            self.rocket_pos + direction, 
            [1, 0], 
            [1, self.grid_height - 1]
        )
        
        # Move all rocks leftward (toward the rocket)
        for i in range(len(self.rocks)):
            self.rocks[i] = self.rocks[i] - np.array([self.current_speed, 0])
        
        # Remove rocks that have moved off the grid
        self.rocks = [rock for rock in self.rocks if rock[0] >= 0]
        
        # Generate new rocks with some probability
        if self.np_random.random() < 0.3:  # 30% chance each step
            self._generate_new_rocks()
        
        # Check for collision
        terminated = False
        for rock in self.rocks:
            # Convert the rock's floating point coordinates to integers for grid-based collision
            rock_pos = np.floor(rock).astype(int)
            if np.array_equal(self.rocket_pos, rock_pos):
                terminated = True
                break
        
        # Increase difficulty (speed) over time
        if self.steps % 20 == 0:  # Every 20 steps
            self.current_speed += self.difficulty_increase_rate
        
        # Reward: +0.1 for surviving each step, -1 for collision
        reward = -1.0 if terminated else 0.1
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Rocket Game")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((0, 0, 0))  # Black background for space
        
        # Calculate square size
        pix_square_size_x = self.window_size[0] / self.grid_width
        pix_square_size_y = self.window_size[1] / self.grid_height
        
        # Draw the rocket (blue triangle)
        # Center the rocket within the grid cell
        rocket_center_x = (self.rocket_pos[0] + 0.5) * pix_square_size_x
        rocket_center_y = (self.rocket_pos[1] + 0.5) * pix_square_size_y
        
        # Make triangle slightly smaller to fit inside cell
        triangle_width = pix_square_size_x * 0.8
        triangle_height = pix_square_size_y * 0.8
        
        points = [
            (rocket_center_x + triangle_width/2, rocket_center_y),  # tip of rocket pointing right
            (rocket_center_x - triangle_width/2, rocket_center_y - triangle_height/2),  # back left
            (rocket_center_x - triangle_width/2, rocket_center_y + triangle_height/2)   # back right
        ]
        pygame.draw.polygon(canvas, (0, 100, 255), points)  # Blue rocket
        
        # Draw the rocks (red squares)
        for rock in self.rocks:
            pygame.draw.rect(
                canvas,
                (255, 50, 50),  # Red rocks
                pygame.Rect(
                    rock[0] * pix_square_size_x,
                    rock[1] * pix_square_size_y,
                    pix_square_size_x,
                    pix_square_size_y
                )
            )
        
        # Draw grid lines
        for x in range(self.grid_width + 1):
            pygame.draw.line(
                canvas,
                (50, 50, 50),  # Dark gray
                (x * pix_square_size_x, 0),
                (x * pix_square_size_x, self.window_size[1]),
                width=1
            )
        for y in range(self.grid_height + 1):
            pygame.draw.line(
                canvas,
                (50, 50, 50),  # Dark gray
                (0, y * pix_square_size_y),
                (self.window_size[0], y * pix_square_size_y),
                width=1
            )
        
        # Add game info text
        font = pygame.font.SysFont(None, 24)
        speed_text = font.render(f"Speed: {self.current_speed:.1f}", True, (255, 255, 255))
        canvas.blit(speed_text, (10, 10))
        
        steps_text = font.render(f"Steps: {self.steps}", True, (255, 255, 255))
        canvas.blit(steps_text, (10, 40))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()