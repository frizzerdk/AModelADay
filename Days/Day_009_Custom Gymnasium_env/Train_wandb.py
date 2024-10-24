import MyUtils.Util.Misc as util
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import argparse
import torch
import os
import sys
from typing import Any
from QuadbotEnv import QuadbotEnv
from gymnasium.wrappers import RecordVideo
from wandb.integration.sb3 import WandbCallback

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

        # Get the current state
        current_state = self.training_env.envs[0].unwrapped.get_state()

        # Get the reward breakdown
        reward_breakdown = self.locals['infos'][0].get('reward_breakdown', {})

        # Log info about the current step
        log_data = {
            "timesteps": self.num_timesteps,
            "current_episode_reward": self._current_episode_reward,
            "current_episode_length": self._current_episode_length,
        }

        # Add state information to log_data
        for key, value in current_state.items():
            log_data[f"state/{key}"] = value

        # Add reward breakdown to log_data
        for key, value in reward_breakdown.items():
            log_data[f"reward/{key}"] = value

        # Reset episode stats if the episode has ended
        if self.locals['dones'][0]:
            log_data["episode_reward"] = self._current_episode_reward
            log_data["episode_length"] = self._current_episode_length
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

        wandb.log(log_data)

        return True

class WandbEvalCallback(BaseCallback):
    def __init__(self, verbose=1):
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
    cfg = util.load_and_override_config(".", "quadbot_config", init_wandb=True, update_wandb=True)
    print(OmegaConf.to_yaml(cfg))

    # Initialize WandB
    wandb.gym.monitor()  # Enable video logging
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
    env = QuadbotEnv(max_steps=cfg.env.max_steps, config=cfg)
    env = Monitor(env, cfg.monitor_dir)

    eval_env = QuadbotEnv(max_steps=cfg.env.max_steps, render_mode="rgb_array", config=cfg)
    eval_env = Monitor(eval_env, cfg.monitor_dir)
    eval_env = RecordVideo(eval_env, video_folder=cfg.video_dir, episode_trigger=lambda x: x % cfg.train.n_eval_episodes == 0)

    # Set up the callbacks
    wandb_callback = WandbCallback()
    wandb_eval_callback = WandbEvalCallback()
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=cfg.checkpoint_path,
        log_path=cfg.log_dir,
        eval_freq=cfg.train.eval_freq,  
        deterministic=True,
        render=cfg.train.render_eval,
        n_eval_episodes=cfg.train.n_eval_episodes,
        callback_after_eval=wandb_eval_callback
    )

    callbacks = [eval_callback, wandb_callback]

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model with GPU support
    model = PPO(cfg.model.policy, env, verbose=1, 
                # learning_rate=cfg.model.learning_rate, 
                # n_steps=cfg.model.n_steps,
                # batch_size=cfg.model.batch_size,
                # n_epochs=cfg.model.n_epochs,
                # gamma=cfg.model.gamma,
                # gae_lambda=cfg.model.gae_lambda,
                # clip_range=cfg.model.clip_range,
                # ent_coef=cfg.model.ent_coef,
                # vf_coef=cfg.model.vf_coef,
                # max_grad_norm=cfg.model.max_grad_norm,
                # use_sde=cfg.model.use_sde,
                
                device=device)

    # Train the model
    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callbacks,
        log_interval=cfg.train.log_interval
    )

    # Save the final model
    model.save(f"{cfg.save.model_dir}/final_model")

    # Log the best model as an artifact
    artifact = wandb.Artifact('best-model', type='model')
    artifact.add_file(f"{cfg.checkpoint_path}/best_model.zip")
    wandb.log_artifact(artifact)

    env.close()
    eval_env.close()
    wandb.finish()

if __name__ == "__main__":
    train()