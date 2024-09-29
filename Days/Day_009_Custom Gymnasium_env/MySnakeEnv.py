import gymnasium as gym
import cv2
import numpy as np
from gymnasium import spaces

class SnakeEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode=None, max_steps=100, grid_size=5):
        self.grid_size = grid_size
        self.window_size = 500  # The size of the PyGame window
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Right, 2: Down, 3: Left
        self.observation_space = spaces.Box(low=-1, high=self.grid_size**2, shape=(self.grid_size, self.grid_size), dtype=int)
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1
        current_head = self.snake[0]
        
        if action == 0:  # Up
            new_head = (current_head[0] - 1, current_head[1])
        elif action == 1:  # Right
            new_head = (current_head[0], current_head[1] + 1)
        elif action == 2:  # Down
            new_head = (current_head[0] + 1, current_head[1])
        else:  # Left
            new_head = (current_head[0], current_head[1] - 1)
        
        # Check if the snake has hit the wall or itself
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size or
            new_head in self.snake):
            return self._get_obs(), -1, True, False, {}
        
        self.snake.insert(0, new_head)
        
        reward = 0
        # Check if the snake has eaten the food
        if new_head == self.food:
            self.score += 1
            self.food = self._place_food()
            reward = 1
        else:
            self.snake.pop()
        
        # Check if max steps reached
        if self.steps >= self.max_steps:
            return self._get_obs(), reward, False, True, {}
        
        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i, segment in enumerate(self.snake):
            obs[segment] = i + 1
        obs[self.food] = -1
        return obs

    def _place_food(self):
        while True:
            food = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if food not in self.snake:
                return food

    def render(self):
        if self.render_mode is None:
            return None
        
        return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            cv2.namedWindow("Snake Game", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Snake Game", self.window_size, self.window_size)

        obs = self._get_obs()
        
        # Normalize the observation
        obs_normalized = np.zeros_like(obs, dtype=float)
        obs_normalized[obs > 0] = obs[obs > 0] / obs[obs > 0].max()
        obs_normalized[obs < 0] = -obs[obs < 0] / obs[obs < 0].min()

        # Color mapping
        canvas = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        canvas[obs == 0] = [255, 255, 255]  # Empty: White
        canvas[obs < 0] = [0, 0, 255]  # Food: Red
        
        # Snake: Green gradient
        snake_mask = obs > 0
        canvas[snake_mask, 1] = np.uint8(obs_normalized[snake_mask] * 255)
        
        # Resize the image to window_size
        canvas = cv2.resize(canvas, (self.window_size, self.window_size), interpolation=cv2.INTER_NEAREST)
        
        # Add text for score and steps
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, f"Score: {self.score}", (10, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, f"Steps: {self.steps}/{self.max_steps}", (10, 60), font, 0.7, (0, 0, 0), 2)

        if self.render_mode == "human":
            cv2.imshow("Snake Game", canvas)
            cv2.waitKey(1)
        
        return canvas

    def close(self):
        if self.window is not None:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    env = SnakeEnv(render_mode="human")
    obs, _ = env.reset()

    print("Snake Game")
    print("Controls: W: Up, D: Right, S: Down, A: Left, Q: Quit")

    while True:
        env.render()
        key = cv2.waitKey(200) & 0xFF  # Wait for 200ms for a key press

        if key == ord('q'):
            break
        elif key == ord('w'):
            action = 0  # Up
        elif key == ord('d'):
            action = 1  # Right
        elif key == ord('s'):
            action = 2  # Down
        elif key == ord('a'):
            action = 3  # Left
        else:
            continue  # If any other key is pressed, continue without taking action

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {env.score}")
            cv2.waitKey(2000)  # Wait for 2 seconds before closing
            break

    env.close()
