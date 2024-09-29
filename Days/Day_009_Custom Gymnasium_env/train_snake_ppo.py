import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from MySnakeEnv import SnakeEnv
import time
import argparse

class RenderEvaluationCallback(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=10, deterministic=True, render_freq=1, render_delay=0.01):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.render_freq = render_freq
        self.render_delay = render_delay

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            self.evaluate_and_render()
        return True

    def evaluate_and_render(self):
        episode_rewards = []
        for episode in range(self.n_eval_episodes):
            print(f"\nEvaluation Episode {episode + 1}/{self.n_eval_episodes}")
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                self.eval_env.render()
                print(f"Action: {action}, Reward: {reward}", end="\r")
                if self.render_delay > 0:
                    time.sleep(self.render_delay)  # Adjustable delay for rendering
            episode_rewards.append(episode_reward)
            print(f"Episode {episode + 1} reward: {episode_reward}")
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        print(f"\nMean reward over {self.n_eval_episodes} episodes: {mean_reward:.2f}")

def main(mode):
    # Create the environment
    env = SnakeEnv(max_steps=100)  # Set max_steps to 100
    eval_env = SnakeEnv(max_steps=100, render_mode="human")

    # Wrap the environment in a DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])

    if mode == 'train':
        # Initialize the PPO agent
        model = PPO("MlpPolicy", vec_env, verbose=1)

        # Create the evaluation callback
        eval_callback = RenderEvaluationCallback(eval_env, n_eval_episodes=2, render_freq=10000, deterministic=False, render_delay=0.02)

        # Train the agent
        total_timesteps = 1000000
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        # Save the trained model
        model.save("ppo_snake_model")

    elif mode == 'evaluate':
        # Load the best model
        model = PPO.load("ppo_snake_model")

    # Final evaluation
    print("\nFinal Evaluation:")
    eval_callback = RenderEvaluationCallback(eval_env, n_eval_episodes=10, render_freq=1, deterministic=False, render_delay=0.02)
    eval_callback.model = model
    eval_callback.evaluate_and_render()

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the Snake PPO model.")
    parser.add_argument('mode', choices=['train', 'evaluate'], default='train', nargs='?',
                        help="Choose 'train' to train a new model or 'evaluate' to evaluate the best saved model. Default is 'evaluate'.")
    args = parser.parse_args()

    main(args.mode)