import MyUtils.Util.Misc as util
import numpy as np
import wandb
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_video_recorder import VecVideoRecorder
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
import argparse
import torch
import os
import sys
from typing import Any
from QuadChaseEnv import QuadChaseEnv
from gymnasium.wrappers import RecordVideo, NormalizeObservation
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the script directory
os.chdir(script_dir)

assert torch.cuda.is_available(), "No GPU available"

def train():
    # Load configuration
    cfg = util.load_and_override_config(".", "base_config", init_wandb=True, update_wandb=True)
    print(OmegaConf.to_yaml(cfg))

    # Add code to the wandb artifacts
    wandb.save("quadbot.xml")
    wandb.save("base_config.yaml")
    wandb.save("QuadChaseEnv.py")

    # Initialize WandB
    wandb.gym.monitor()  # Enable video logging
    wandb.define_metric("eval/mean_reward", summary="mean")
    wandb.define_metric("eval/mean_ep_length", summary="mean")
    wandb.define_metric("eval/min_ep_reward", summary="mean")
    wandb.define_metric("eval/success_rate", summary="mean")
    wandb.define_metric("episode/*", summary="mean")
    wandb.define_metric("reward/*", summary="mean")

    # Create the environments
    env = DummyVecEnv([lambda: QuadChaseEnv(max_steps=cfg.env.max_steps,
                        config=cfg.env,
                        frame_skip=cfg.env.frame_skip)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True)


    eval_env = DummyVecEnv([lambda: QuadChaseEnv(max_steps=cfg.env.max_steps,
                             config=cfg.env,
                             frame_skip=cfg.env.frame_skip,
                             render_mode="rgb_array")])
    
    eval_env = VecMonitor(eval_env, cfg.monitor_dir)
    eval_env = VecVideoRecorder(eval_env,
                            video_folder=cfg.video_dir, 
                            #episode_trigger=lambda x: x % (cfg.train.video_freq*cfg.train.n_eval_episodes) == 0, 
                            record_video_trigger=lambda x: True,
                            video_length=cfg.train.video_length)

    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)





    # Remove the RecordVideo wrapper from eval_env
    #eval_env = NormalizeObservation(eval_env)

    # Set up the callbacks
    wandb_callback = WandbCallback()
    wandb_eval_callback = WandbEvalCallback()


    video_eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=cfg.checkpoint_path,
        log_path=cfg.log_dir,
        eval_freq=cfg.train.eval_freq*cfg.train.video_freq,  
        deterministic=True,
        render=cfg.train.render_eval,
        n_eval_episodes=1,
        callback_after_eval=wandb_eval_callback,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=cfg.checkpoint_path,
        log_path=cfg.log_dir,
        eval_freq=cfg.train.eval_freq,  
        deterministic=True,
        render=False,
        n_eval_episodes=cfg.train.n_eval_episodes,
        callback_after_eval=wandb_eval_callback,
        verbose=1
    )
    

    callbacks = [video_eval_callback, eval_callback, wandb_callback]

    # Specify the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # custom network
    policy_kwargs = dict(activation_fn=torch.nn.LeakyReLU,
                         net_arch=dict(pi=[cfg.model.hidden_layers_size, cfg.model.hidden_layers_size], vf=[cfg.model.hidden_layers_size, cfg.model.hidden_layers_size]))
    # Load the model with GPU support
    model = PPO(cfg.model.policy, env, verbose=1, policy_kwargs=policy_kwargs,
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
                use_sde=cfg.model.use_sde,
                
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
        log_dict = self.training_env.envs[0].unwrapped.get_log_dict()

        log_data = dict()
        # Log info about the current step
        if self.verbose >= 2:
            log_data = {
                "current/timesteps": self.num_timesteps,
                "current/episode_reward": self._current_episode_reward,
                "current/episode_length": self._current_episode_length,
            }

            # Add state information to log_data
            for key, value in log_dict.items():
                log_data[key] = value

        # Reset episode stats if the episode has ended
        if self.locals['dones'][0]:
            log_data["episode/reward"] = self._current_episode_reward
            log_data["episode/length"] = self._current_episode_length
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

            episode_log_dict = self.training_env.envs[0].unwrapped.get_previous_episode_summary()
            for key, value in episode_log_dict.items():
                log_data[key] = value

        if log_data:
            wandb.log(log_data)

        return True



class WandbEvalCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # This method will be called after each evaluation
        # Log the evaluation results to wandb
        wandb.log({
            "eval/mean_reward": self.parent.last_mean_reward,
            "eval/mean_ep_length": np.mean(self.parent.evaluations_length[-1]) if self.parent.evaluations_length else None,
        })
        
        # If you want to log success rate (if available)
        if self.parent.evaluations_successes:
            success_rate = np.mean(self.parent.evaluations_successes[-1])
            wandb.log({"eval/success_rate": success_rate})
        return True 


if __name__ == "__main__":
    train()
