## 1. Description
RocketGame is a two-dimensional grid environment where a player controls a rocket that moves vertically (up, down, or stays in place) on the left side of the board. The player's goal is to avoid meteoroids that are randomly generated on the right side of the board. The meteoroids move from right to left, and a collision with any of them ends the game.
Over time, the game's difficulty increases—the speed of the meteoroids' movement grows, which forces faster reactions and more advanced avoidance strategies.
The environment was implemented as a custom Gymnasium-compliant environment.

<img width="1085" height="795" alt="image2" src="https://github.com/user-attachments/assets/e774005b-0bf2-4742-8e22-ee8a1fd0e1db" />

For training the agent that controls the rocket, the Deep Q-Network (DQN) was used.
### Key features of the DQN implementation:
- Input: an observation vector (rocket's position + the position of each meteoroid + speed).
- Spatial transformation: the input data is projected onto a two-dimensional map (grid) with separate channels for the rocket's and meteoroids' positions.

## 2. CNN network architecture:
2 convolutional layers,
3 fully connected (FC) layers,
an additional input for speed as a numerical variable.

## 3.Training

The training consisted of iteratively playing episodes and updating the neural network weights based on the collected experiences (Replay Memory). The agent learned to avoid collisions by maximizing the sum of rewards, a positive reward was granted for each survived step, while a large penalty was given for the collision.

<img width="1660" height="877" alt="image3" src="https://github.com/user-attachments/assets/d851a5f1-99c5-48c1-adb0-c04339a46e5c" />


#4. Install dependencies with uv

```
uv init
uv sync
```
