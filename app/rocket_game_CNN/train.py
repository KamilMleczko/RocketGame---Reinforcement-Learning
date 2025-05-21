import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from collections import deque, namedtuple
import random
from rocket_env_rl import RocketEnvRL
import time

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 5000
TARGET_UPDATE = 50
LEARNING_RATE = 0.0001
MEMORY_SIZE = 100000
NUM_EPISODES = 2000
SAVE_INTERVAL = 100
SAVE_DIR = './models'

# Define the CNN model
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
                    if 0 <= rock_x < self.grid_width and 0 <= rock_y < self.grid_height:
                        if rock_x >= 0 and rock_y >= 0:  # Skip -1 values (no rock)
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

# Define replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Create the environment
def create_env(render_mode=None):
    return RocketEnvRL(render_mode=render_mode, grid_height=10, grid_width=15, difficulty_increase_rate=0.05)

# Set up the environment and networks
env = create_env()
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQNCnn(obs_size, n_actions, grid_width=env.grid_width, grid_height=env.grid_height)
target_net = DQNCnn(obs_size, n_actions, grid_width=env.grid_width, grid_height=env.grid_height)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

# Select action using epsilon-greedy policy
def select_action(state, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                   np.exp(-1. * steps_done / EPS_DECAY)
    
    if sample > eps_threshold:
        with torch.no_grad():
            # Make sure state is properly unsqueezed to have batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return policy_net(state_tensor).max(1)[1].item()
    else:
        return random.randrange(n_actions)

# Update the policy network
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Create masks for non-terminal states
    non_final_mask = torch.tensor(tuple(map(lambda s: not s, batch.done)), dtype=torch.bool)
    
    # Convert the list of numpy arrays to a single numpy array first
    next_states = [s for s, d in zip(batch.next_state, batch.done) if not d]
    if next_states:
        non_final_next_states = torch.FloatTensor(np.array(next_states))
    else:
        non_final_next_states = torch.FloatTensor([])
    
    state_batch = torch.FloatTensor(np.array(batch.state))
    action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1)
    reward_batch = torch.FloatTensor(np.array(batch.reward))
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Initialize next state values to zero
    next_state_values = torch.zeros(BATCH_SIZE)
    
    # Compute V(s_{t+1}) for all next states
    if len(non_final_next_states) > 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss.item()

# Create save directory if it doesn't exist
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Training loop
episode_rewards = []
avg_rewards = []
steps_done = 0

print("Starting training with CNN architecture...")
for episode in range(NUM_EPISODES):
    # Initialize the environment and state
    state, _ = env.reset()
    episode_reward = 0
    episode_loss = 0
    num_steps = 0
    
    # Play one episode
    while True:
        # Select and perform an action
        action = select_action(state, steps_done)
        next_state, reward, done, _, _ = env.step(action)
        steps_done += 1
        num_steps += 1
        episode_reward += reward
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward, done)
        
        # Move to the next state
        state = next_state
        
        # Perform one step of the optimization
        loss = optimize_model()
        if loss is not None:
            episode_loss += loss
        
        # Update the target network every TARGET_UPDATE steps
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if done:
            break
    
    # Log episode stats
    episode_rewards.append(episode_reward)
    if len(episode_rewards) >= 10:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_rewards.append(avg_reward)
    else:
        avg_reward = np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
    
    if episode % 10 == 0:
        avg_loss = episode_loss / num_steps if episode_loss > 0 else 0
        eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
        print(f"Episode {episode}/{NUM_EPISODES} | Steps: {num_steps} | Reward: {episode_reward:.2f} | "
              f"Avg Reward: {avg_reward:.2f} | Eps: {eps_threshold:.2f} | Avg Loss: {avg_loss:.4f}")
    
    # Save model checkpoint
    if episode % SAVE_INTERVAL == 0 or episode == NUM_EPISODES - 1:
        torch.save({
            'policy_net': policy_net.state_dict(),
            'target_net': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'episode': episode,
            'steps_done': steps_done,
        }, f"{SAVE_DIR}/rocket_dqn_cnn_episode_{episode}.pt")

# Plot the training progress
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 2, 2)
plt.plot(avg_rewards)
plt.title('Average Rewards (over 10 episodes)')
plt.xlabel('Episode')
plt.ylabel('Avg Reward')

plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/training_progress_cnn.png")

# Save the final model
torch.save({
    'policy_net': policy_net.state_dict(),
    'target_net': target_net.state_dict(),
    'optimizer': optimizer.state_dict(),
    'episode': NUM_EPISODES - 1,
    'steps_done': steps_done,
}, f"{SAVE_DIR}/rocket_dqn_cnn_final.pt")

print("Training complete!")
env.close()