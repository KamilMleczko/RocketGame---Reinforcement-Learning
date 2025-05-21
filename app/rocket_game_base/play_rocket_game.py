import pygame
import numpy as np
import sys
from env.rocket_env import RocketEnv, Actions

def play_rocket_game():
    # Initialize the environment
    env = RocketEnv(render_mode="human", grid_height=10, grid_width=15)
    observation, info = env.reset()
    
    # Set up pygame
    pygame.init()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)
    
    # Game state variables
    running = True
    game_over = False
    total_reward = 0
    
    # Control cooldown to prevent too rapid movement
    action_cooldown = 0
    COOLDOWN_TIME = 0 # frames to wait between actions
    
    print("\nRocket Game Controls:")
    print("  UP ARROW: Move rocket up")
    print("  DOWN ARROW: Move rocket down")
    print("  SPACE: Stay in place")
    print("  R: Reset game")
    print("  Q or ESC: Quit game")
    print("\nAvoid the red rocks! Your rocket stays in one column, but you can move up and down.")
    print("The game gets faster over time. Good luck!\n")
    
    # Main game loop
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                    
                # Reset game when 'R' is pressed
                if event.key == pygame.K_r:
                    observation, info = env.reset()
                    game_over = False
                    total_reward = 0
        
        # If game is not over, process player input
        if not game_over:
            # Decrement cooldown if it's active
            if action_cooldown > 0:
                action_cooldown -= 1
            
            # Get keyboard state and determine action
            keys = pygame.key.get_pressed()
            action = Actions.stay.value  # Default: stay in place
            
            # Only process input if cooldown is done
            if action_cooldown == 0:
                if keys[pygame.K_UP]:
                    action = Actions.up.value
                    action_cooldown = COOLDOWN_TIME
                elif keys[pygame.K_DOWN]:
                    action = Actions.down.value
                    action_cooldown = COOLDOWN_TIME
            
            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Check if game is over
            if terminated:
                game_over = True
                
                # Draw game over text
                if env.window is not None:
                    game_over_text = font.render(f"GAME OVER! Score: {info['score']}", True, (255, 0, 0))
                    env.window.blit(game_over_text, 
                                    (env.window_size[0]//4, env.window_size[1]//2))
                    restart_text = font.render("Press 'R' to restart", True, (255, 255, 255))
                    env.window.blit(restart_text, 
                                    (env.window_size[0]//4, env.window_size[1]//2 + 40))
                    pygame.display.update()
        
        # Cap the frame rate
        clock.tick(30)
    
    # Clean up
    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    play_rocket_game()
