import MyUtils.Util.Misc as util
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import gymnasium as gym
import stable_baselines3 as sb3
import argparse
import torch
import os
import sys
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

assert torch.cuda.is_available(), "No GPU available"

class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandbCallback, self).__init__(verbose)
    
    def _on_step(self) -> bool:

        wandb.log({
            "timesteps": self.num_timesteps,
            "episode_reward": self.locals['ep_info_buffer'][-1]['r'] if len(self.locals['ep_info_buffer']) > 0 else None,
            "episode_length": self.locals['ep_info_buffer'][-1]['l'] if len(self.locals['ep_info_buffer']) > 0 else None,
        })
        return True


def train():
    # Load configuration
    cfg = util.load_and_override_config(".", "config",init_wandb=True,update_wandb=True)
    print(OmegaConf.to_yaml(cfg))

    # Initialize WandB
    wandb.define_metric("epoch/eval_reward", summary="max")
    wandb.define_metric("epoch/eval_reward", summary="mean")    

    
    # Load the environment
    env = gym.make(cfg.env_name)
    env = sb3.common.monitor.Monitor(env, cfg.monitor_dir)
    env = sb3.common.vec_env.DummyVecEnv([lambda: env])

    # Load the evaluation environment
    eval_env = gym.make(cfg.env_name)
    eval_env = sb3.common.monitor.Monitor(eval_env, cfg.monitor_dir)
    eval_env = sb3.common.vec_env.DummyVecEnv([lambda: eval_env])

    # Set up the evaluation callback
    eval_callback = EvalCallback(eval_env, best_model_save_path=cfg.best_model_path, log_path=cfg.log_dir, eval_freq=cfg.eval_freq, deterministic=True, render=False, verbose=1)
    wandb_callback = WandbCallback()

    # Load the model
    model = sb3.PPO(cfg.model, env, verbose=1, learning_rate=cfg.learning_rate, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range)

    # Train the model
    model.learn(total_timesteps=cfg.total_timesteps, callback=[eval_callback, wandb_callback])


    # Log the best model as an artifact
    artifact = wandb.Artifact('best-model', type='model')
    artifact.add_file(cfg.best_model_path)
    wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    train()

