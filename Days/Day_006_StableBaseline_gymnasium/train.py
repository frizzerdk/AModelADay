import MyUtils.Util.Misc as util
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import stable_baselines3 as sb3
import argparse
import torch
import os
import sys
from typing import Any

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
    wandb.gym.monitor()
    wandb.define_metric("eval/mean_reward")
    wandb.define_metric("eval/mean_ep_length")
    wandb.define_metric("eval/max_ep_reward")
    wandb.define_metric("eval/min_ep_reward")
    wandb.define_metric("eval/success_rate")

    # Load the training environment
    env = gym.make(cfg.env_name, render_mode="rgb_array")
    env = sb3.common.monitor.Monitor(env, cfg.monitor_dir)

    # Load the evaluation environment
    eval_env = gym.make(cfg.env_name, render_mode="rgb_array")
    eval_env = sb3.common.monitor.Monitor(eval_env, cfg.monitor_dir)
    eval_env = RecordVideo(eval_env, video_folder=cfg.video_dir, episode_trigger=lambda x: x % cfg.n_eval_episodes == 0)
    eval_env = sb3.common.vec_env.DummyVecEnv([lambda: eval_env])

    # Set up the callbacks
    wandb_callback = WandbCallback()
    wandb_eval_callback = WandbEvalCallback()
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=cfg.checkpoint_path,
        log_path=cfg.log_dir,
        eval_freq=cfg.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=cfg.n_eval_episodes,
        callback_after_eval=wandb_eval_callback
    )

    callbacks = [eval_callback, wandb_callback]

    # Load the model
    model = sb3.PPO(cfg.model, env, verbose=1, learning_rate=cfg.learning_rate, 
                    gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range)

    # Train the model
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks
    )

    # Log the best model as an artifact
    # Convert cfg.best_model_path to an absolute path
    # artifact = wandb.Artifact('best-model', type='model')
    # artifact.add_file(cfg.best_model_path)
    # wandb.log_artifact(artifact)
    env.close()
    eval_env.close()
    wandb.finish()

if __name__ == "__main__":
    train()
