import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rocket_env_rl import RocketEnvRL
import pygame
import argparse
import os
import time

# Define the CNN model (must match the one used in training)
class DQNCnn(nn.Module):
    def __init__(self, input_size, output_size, grid_width=15, grid_height=10):
        super(DQNCnn, self).__init__()
        
        # Convert flat input to 2D grid + features
        self.input_size = input_size
        self.grid_width = grid_width
        self.grid_height = grid_height
        
        # CNN layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = 32 * grid_width * grid_height
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + 1, 256)  # +1 for speed
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract speed (last element)
        speed = x[:, -1].unsqueeze(1)
        
        # Process the rest of the input to create a spatial representation
        # Format: [rocket_y, rock1_x, rock1_y, rock2_x, rock2_y, ...]
        
        # Create empty grids (2 channels: rocket channel and rock channel)
        rocket_grid = torch.zeros(batch_size, 1, self.grid_height, self.grid_width, device=x.device)
        rock_grid = torch.zeros(batch_size, 1, self.grid_height, self.grid_width, device=x.device)
        
        for i in range(batch_size):
            # Place rocket (first element is rocket_y)
            rocket_y = int(x[i, 0].item())
            if 0 <= rocket_y < self.grid_height:
                rocket_grid[i, 0, rocket_y, 1] = 1.0  # Rocket at x=1
            
            # Place rocks (pairs of x,y coordinates)
            for j in range(1, len(x[i])-1, 2):
                if j+1 < len(x[i]):
                    rock_x = int(x[i, j].item())
                    rock_y = int(x[i, j+1].item())
                    if rock_x >= 0 and rock_y >= 0:  # Skip -1 values (no rock)
                        if 0 <= rock_x < self.grid_width and 0 <= rock_y < self.grid_height:
                            rock_grid[i, 0, rock_y, rock_x] = 1.0
        
        # Combine channels
        grid = torch.cat([rocket_grid, rock_grid], dim=1)
        
        # Apply convolutions
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Concatenate with speed
        x = torch.cat([x, speed], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def main():
    parser = argparse.ArgumentParser(description='Watch the trained CNN agent play Rocket Game')
    parser.add_argument('--model', type=str, default='./models/rocket_dqn_cnn_early_stop_600.pt',
                        help='path to the trained model file')
    parser.add_argument('--grid_height', type=int, default=10, help='grid height')
    parser.add_argument('--grid_width', type=int, default=15, help='grid width')
    parser.add_argument('--episodes', type=int, default=5, help='number of episodes to play')
    parser.add_argument('--difficulty', type=float, default=0.05, 
                        help='difficulty increase rate (how fast the game speeds up)')
    parser.add_argument('--fps', type=int, default=10, help='frames per second for rendering')
    parser.add_argument('--delay', type=float, default=0.0, 
                        help='additional delay between steps (seconds)')
    args = parser.parse_args()
    
    # Ensure model file exists
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found!")
        return
    
    # Create environment
    env = RocketEnvRL(
        render_mode="human", 
        grid_height=args.grid_height, 
        grid_width=args.grid_width,
        difficulty_increase_rate=args.difficulty
    )
    # Override FPS for better visualization
    env.metadata["render_fps"] = args.fps
    
    # Load the trained model
    checkpoint = torch.load(args.model)
    
    # Create the DQN model with correct input/output sizes
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    model = DQNCnn(obs_size, n_actions, grid_width=env.grid_width, grid_height=env.grid_height)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['policy_net'])
    model.eval()  # Set to evaluation mode
    
    print(f"Loaded CNN model from {args.model}")
    print(f"Playing {args.episodes} episodes with FPS={args.fps}...")
    
    # Track statistics over all episodes
    all_scores = []
    all_steps = []
    
    for episode in range(args.episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"Starting episode {episode+1}/{args.episodes}")
        
        while True:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:  # Press 'Q' to quit
                        env.close()
                        pygame.quit()
                        return
                    
                    if event.key == pygame.K_n:  # Press 'N' to skip to next episode
                        break
            
            # Use the model to select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = model(state_tensor).max(1)[1].item()
            
            # Take the action
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Move to the next state
            state = next_state
            
            # Add additional delay if specified
            if args.delay > 0:
                time.sleep(args.delay)
            
            # Check if episode is over
            if done:
                print(f"Episode {episode+1} finished | Steps: {steps} | Score: {info['score']} | Final speed: {info['speed']:.2f}")
                all_scores.append(info['score'])
                all_steps.append(steps)
                break
    
    # Display overall statistics
    if all_scores:
        print("\nOverall Statistics:")
        print(f"Average Score: {np.mean(all_scores):.1f}")
        print(f"Average Steps: {np.mean(all_steps):.1f}")
        print(f"Best Score: {np.max(all_scores)}")
        print(f"Best Steps: {np.max(all_steps)}")
    
    # Close environment
    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()