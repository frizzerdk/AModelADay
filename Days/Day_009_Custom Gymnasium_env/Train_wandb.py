import MyUtils.Util.Misc as util
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import torch
import os
import sys
from typing import Any
from MySnakeEnv import SnakeEnv
from gymnasium.wrappers import RecordVideo

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

assert torch.cuda.is_available(), "No GPU available"

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Update current episode stats
        self._current_episode_reward += self.locals['rewards'][0]
        self._current_episode_length += 1

        # Log info about the current step
        log_data = {
            "timesteps": self.num_timesteps,
            "current_episode_reward": self._current_episode_reward,
            "current_episode_length": self._current_episode_length,
        }

        # Reset episode stats if the episode has ended
        if self.locals['dones'][0]:
            log_data["episode_reward"] = self._current_episode_reward
            log_data["episode_length"] = self._current_episode_length
            wandb.log(log_data)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
        else:
            wandb.log(log_data)

        return True

class WandbEvalCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # This method will be called after each evaluation
        # Log the evaluation results to wandb
        wandb.log({
            "eval/timesteps": self.parent.num_timesteps,
            "eval/mean_reward": self.parent.last_mean_reward,
            "eval/mean_ep_length": np.mean(self.parent.evaluations_length[-1]) if self.parent.evaluations_length else None,
        })
        
        # If you want to log success rate (if available)
        if self.parent.evaluations_successes:
            success_rate = np.mean(self.parent.evaluations_successes[-1])
            wandb.log({"eval/success_rate": success_rate})
        return True 

def train():
    # Load configuration
    cfg = util.load_and_override_config(".", "config", init_wandb=True, update_wandb=True)
    print(OmegaConf.to_yaml(cfg))

    # Initialize WandB
    wandb.define_metric("eval/mean_reward", summary="min")
    wandb.define_metric("eval/mean_reward", summary="max")
    wandb.define_metric("eval/mean_reward", summary="mean")
    wandb.define_metric("eval/mean_ep_length", summary="min")
    wandb.define_metric("eval/mean_ep_length", summary="max")
    wandb.define_metric("eval/mean_ep_length", summary="mean")
    wandb.define_metric("eval/max_ep_reward", summary="min")
    wandb.define_metric("eval/max_ep_reward", summary="max")
    wandb.define_metric("eval/max_ep_reward", summary="mean")
    wandb.define_metric("eval/min_ep_reward", summary="min")
    wandb.define_metric("eval/min_ep_reward", summary="max")
    wandb.define_metric("eval/min_ep_reward", summary="mean")
    wandb.define_metric("eval/success_rate", summary="min")
    wandb.define_metric("eval/success_rate", summary="max")
    wandb.define_metric("eval/success_rate", summary="mean")

    # Create the environments
    env = SnakeEnv(max_steps=cfg.env.max_steps, grid_size=cfg.env.grid_size)
    
    eval_env = SnakeEnv(max_steps=cfg.env.max_steps, grid_size=cfg.env.grid_size, render_mode="rgb_array")
    eval_env = RecordVideo(eval_env, video_folder=cfg.video_dir, episode_trigger=lambda x: x % cfg.n_eval_episodes == 0)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Set up the callbacks
    wandb_callback = WandbCallback()
    wandb_eval_callback = WandbEvalCallback()
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=cfg.checkpoint_path,
        log_path=cfg.log_dir,
        eval_freq=cfg.train.eval_freq,  # Change this line
        deterministic=True,
        render=False,
        n_eval_episodes=cfg.train.n_eval_episodes,
        callback_after_eval=wandb_eval_callback
    )

    callbacks = [eval_callback, wandb_callback]

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model with GPU support
    model = PPO(cfg.model.policy, env, verbose=1, 
                learning_rate=cfg.model.learning_rate, 
                n_steps=cfg.model.n_steps,
                batch_size=cfg.model.batch_size,
                n_epochs=cfg.model.n_epochs,
                gamma=cfg.model.gamma,
                gae_lambda=cfg.model.gae_lambda,
                clip_range=cfg.model.clip_range,
                ent_coef=cfg.model.ent_coef,
                vf_coef=cfg.model.vf_coef,
                max_grad_norm=cfg.model.max_grad_norm,
                device=device)

    # Train the model
    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callbacks,
        log_interval=cfg.train.log_interval
    )

    # Save the final model
    model.save(f"{cfg.save.model_dir}/final_model")

    env.close()
    eval_env.close()
    wandb.finish()

if __name__ == "__main__":
    train()