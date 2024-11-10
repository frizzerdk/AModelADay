import numpy as np
from gymnasium import utils
from mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import os
import mujoco
from pynput import keyboard
import time
from omegaconf import OmegaConf, omegaconf
from gymnasium import error, spaces
from gymnasium.spaces import Space
import util
from collections import defaultdict
import time

import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io

DEFAULT_CAMERA_CONFIG = {
    "distance": 10,
    "lookat": np.array([0.0, -1, 0.5]),  # x, y, z coordinates
}

class QuadChaseEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"]
    }

    def __init__(self, 
                 xml_file : str = "quadbot.xml", 
                 frame_skip : int = 5, 
                 default_camera_config : dict = DEFAULT_CAMERA_CONFIG, 
                 config= None,
                 max_steps : int = 1000,
                 **kwargs
                 ):
        

        # init local variables
        self._max_steps = max_steps
        self._steps = 0

        utils.EzPickle.__init__(self, xml_file, frame_skip, default_camera_config, config, **kwargs)
        # Set default config, handle dict, None and OmegaConf
        if config is None:
            self.config = OmegaConf.create()
        elif isinstance(config, dict):
            self.config = OmegaConf.create(config)   
        elif isinstance(config, omegaconf.DictConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        ############################################################
        # Set default values for config
        # Logging
        self.config.log_state = self.config.get("log_state", False)
        self.config.log_sensor = self.config.get("log_sensor", False)
        self.config.log_ctrl = self.config.get("log_ctrl", True)
        self.config.log_reward = self.config.get("log_reward", True)
        self.config.log_observation = self.config.get("log_observation", True)
        # Robot physical properties
        self.config.steer_joint_damping = self.config.get("steer_joint_damping", 0.1)
        self.config.drive_joint_damping = self.config.get("drive_joint_damping", 0.05)
        self.config.steer_actuator_gain = self.config.get("steer_actuator_gain", 3)
        self.config.drive_actuator_gain = self.config.get("drive_actuator_gain", 5)
        self.config.steer_kp = self.config.get("steer_kp", 10)

        # Observation space parameters
        self.config.observation_max_wheel_steer_vel = self.config.get("observation_max_wheel_steer_vel", 5)
        self.config.observation_max_wheel_drive_vel = self.config.get("observation_max_wheel_drive_vel", 100)
        self.config.observation_max_body_linear_vel = self.config.get("observation_max_body_linear_vel", 20)
        self.config.observation_max_body_angular_vel = self.config.get("observation_max_body_angular_vel", 3)
        self.config.observation_max_body_position = self.config.get("observation_max_body_position", 10)     
        self.config.unit_ranges = self.config.get("unit_ranges", True)

        # Environment reset parameters
        self.config.reset_noise_scale = self.config.get("reset_noise_scale", 1)
        self.config.reset_pos_offset = self.config.get("reset_pos_offset", 2)
        self.config.target_pos_distance = self.config.get("target_pos_distance", 3)

        # Reward function parameters
        self.config.reward_scale_action_penalty = self.config.get("reward_scale_action_penalty", 0.1)
        self.config.reward_scale_wheel_misalignment = self.config.get("reward_scale_wheel_misalignment", 1.0)
        self.config.reward_scale_position_penalty = self.config.get("reward_scale_position_penalty", 0.0)
        self.config.reward_scale_success_reward = self.config.get("reward_scale_success_reward", 100)
        self.config.reward_scale_position_error_change = self.config.get("reward_scale_position_error_change", 0.0)
        self.config.reward_scale_distance_closed_to_target = self.config.get("reward_scale_distance_closed_to_target", 0.0)
        self.config.reward_scale_total = self.config.get("reward_scale_total", 1.0)
        self.config.reward_scale_position_error_velocity = self.config.get("reward_scale_position_error_velocity", 10.0)
        self.config.reward_scale_drive_action_penalty = self.config.get("reward_scale_drive_action_penalty", 1)
        self.config.reward_scale_steer_action_penalty = self.config.get("reward_scale_steer_action_penalty", 1)

        # Other parameters
        self.config.wheel_alignment_threshold = self.config.get("wheel_alignment_threshold", 10)
        ############################################################
        # State space mapping
        self.qpos_map = {
            "body_pos_x": 0,
            "body_pos_y": 1,
            "body_pos_z": 2,
            "body_quaternion_w": 3,
            "body_quaternion_x": 4,
            "body_quaternion_y": 5,
            "body_quaternion_z": 6,
            "drive_BL": 15,
            "drive_BR": 18,
            "drive_FL": 9,
            "drive_FR": 12,
            "slide_BL": 13,
            "slide_BR": 16,
            "slide_FL": 7,
            "slide_FR": 10,
            "steer_BL": 14,
            "steer_BR": 17,
            "steer_FL": 8,
            "steer_FR": 11
        }
        self.qvel_map = {
            "body_linear_vel_x": 0,
            "body_linear_vel_y": 1,
            "body_linear_vel_z": 2,
            "body_angular_vel_x": 3,
            "body_angular_vel_y": 4,
            "body_angular_vel_z": 5,
            "slide_FL": 6,
            "slide_FR": 9,
            "slide_BL": 12,
            "slide_BR": 15,
            "steer_FL": 7,
            "steer_FR": 10,
            "steer_BL": 13,
            "steer_BR": 16,
            "drive_FL": 8,
            "drive_FR": 11,
            "drive_BL": 14,
            "drive_BR": 17
        }
        self.sensor_map = {
            "steerpos_FL": 0, # angles of wheel steer positions (4)
            "steerpos_FR": 1,
            "steerpos_BL": 2,
            "steerpos_BR": 3,
            "steervel_FL": 4, # angular velocities of wheel steer positions (4)
            "steervel_FR": 5,
            "steervel_BL": 6,
            "steervel_BR": 7,
            "drivevel_FL": 8, # wheel drive velocities (4)
            "drivevel_FR": 9,
            "drivevel_BL": 10,
            "drivevel_BR": 11,
            "body_pos_x": 12, # body position (3)
            "body_pos_y": 13,
            "body_pos_z": 14,
            "body_linvel_x": 15, # body linear velocity (3)
            "body_linvel_y": 16,
            "body_linvel_z": 17,
            "body_angvel_x": 18, # body angular velocity (3)
            "body_angvel_y": 19,
            "body_angvel_z": 20,
            "body_xaxis_1": 21, # body x-axis orientation (3)
            "body_xaxis_2": 22,
            "body_xaxis_3": 23
        }
        self.ctrl_map = {
            "motor_FL_drive": 0,
            "motor_FR_drive": 1,
            "motor_BL_drive": 2,
            "motor_BR_drive": 3,
            "motor_FL_steer": 4,
            "motor_FR_steer": 5,
            "motor_BL_steer": 6,
            "motor_BR_steer": 7
        }
        # Define observation space
        self.observation_map = {
            "steer_FL_cos": 0, # cos of wheel steer positions (4)
            "steer_FR_cos": 1,
            "steer_BL_cos": 2,
            "steer_BR_cos": 3,
            "steer_FL_sin": 4, # sin of wheel steer positions (4)
            "steer_FR_sin": 5,
            "steer_BL_sin": 6,
            "steer_BR_sin": 7,
            "steer_FL_vel": 8, # wheel steer velocities (4)
            "steer_FR_vel": 9,
            "steer_BL_vel": 10,
            "steer_BR_vel": 11,
            "drive_FL_vel": 12, # wheel drive velocities (4)
            "drive_FR_vel": 13,
            "drive_BL_vel": 14,
            "drive_BR_vel": 15,
            "body_linvel_x": 16, # body linear velocity (3)
            "body_linvel_y": 17,
            "body_linvel_z": 18,
            "body_angvel_x": 19, # body angular velocity (3)
            "body_angvel_y": 20,
            "body_angvel_z": 21,
            "body_xaxis_1": 22, # body x-axis orientation (3)
            "body_xaxis_2": 23,
            "body_xaxis_3": 24,
            "body_pos_x": 25, # body position (2)
            "body_pos_y": 26
        }
        obs_max = np.array([
            *np.ones(4)*1,  # cos of wheel steer positions (4)
            *np.ones(4)*1,  # sin of wheel steer positions (4)
            *np.ones(4)*self.config.observation_max_wheel_steer_vel,  # wheel steer velocities (4)
            *np.ones(4)*self.config.observation_max_wheel_drive_vel,  # wheel drive velocities (4)
            *np.ones(3)*self.config.observation_max_body_linear_vel,  # body linear velocity (3)
            *np.ones(3)*self.config.observation_max_body_angular_vel,  # body angular velocity (3)
            *np.ones(3)*1,  # body x-axis orientation (3)
            *np.ones(2)*self.config.observation_max_body_position  # body position (2)
        ])
        obs_min = -obs_max
        if self.config.unit_ranges:
            self.observation_space = Box(low=-np.ones_like(obs_min), high=np.ones_like(obs_max), dtype=np.float64)
            self.observation_space_raw = Box(low=obs_min, high=obs_max, dtype=np.float64)
        else:
            self.observation_space = Box(low=obs_min, high=obs_max, dtype=np.float64)
            self.observation_space_raw = self.observation_space
        # Make a temporary xml file with modified parameters
        file_dir = os.path.dirname(os.path.abspath(__file__))
        temp_xml_file = f"{xml_file}_temp_{random.randint(0, 1000000)}.xml"
        temp_xml_file = os.path.join(file_dir, temp_xml_file)
        
        # target
        self.target_pos = np.array([self.config.target_pos_distance, self.config.target_pos_distance])
        

        util.make_modified_xml_file( os.path.join(file_dir, xml_file), temp_xml_file, 
                                    [("steer_BR_joint", "damping", self.config.steer_joint_damping),
                                     ("steer_BL_joint", "damping", self.config.steer_joint_damping),
                                     ("steer_FR_joint", "damping", self.config.steer_joint_damping),
                                     ("steer_FL_joint", "damping", self.config.steer_joint_damping),
                                     ("drive_BR_joint", "damping", self.config.drive_joint_damping),
                                     ("drive_BL_joint", "damping", self.config.drive_joint_damping),
                                     ("drive_FR_joint", "damping", self.config.drive_joint_damping),
                                     ("drive_FL_joint", "damping", self.config.drive_joint_damping),
                                     ("motor_BR_drive", "gear", self.config.drive_actuator_gain),
                                     ("motor_BL_drive", "gear", self.config.drive_actuator_gain),
                                     ("motor_FR_drive", "gear", self.config.drive_actuator_gain),
                                     ("motor_FL_drive", "gear", self.config.drive_actuator_gain),
                                     ("motor_BR_steer", "gear", self.config.steer_actuator_gain),
                                     ("motor_BL_steer", "gear", self.config.steer_actuator_gain),
                                     ("motor_FR_steer", "gear", self.config.steer_actuator_gain),
                                     ("motor_FL_steer", "gear", self.config.steer_actuator_gain),
                                     ("centre", "pos", f"{self.target_pos[0]} {self.target_pos[1]} 0.001"),
                                     ("motor_FL_steer", "kp", self.config.steer_kp),
                                     ("motor_FR_steer", "kp", self.config.steer_kp),
                                     ("motor_BL_steer", "kp", self.config.steer_kp),
                                     ("motor_BR_steer", "kp", self.config.steer_kp)])
        
        MujocoEnv.__init__(self,
                           temp_xml_file, 
                           frame_skip, 
                           observation_space=self.observation_space, 
                           default_camera_config=default_camera_config, 
                           **kwargs)

        os.remove(temp_xml_file)
        # render fps
        #self.metadata["render_fps"] = int(1.0 / (self.model.opt.timestep * frame_skip))

        self.current_episode_data = defaultdict(list)
        self.previous_episode_data = defaultdict(list)
        self.episode_count = 0
        
        self.reset_model()
        
        
    
    def step(self, action):
        self._steps += 1
        if self.config.unit_ranges:
            action_range = self.action_space_raw.high - self.action_space_raw.low
            action_center = (self.action_space_raw.high + self.action_space_raw.low) / 2
            action = action_center + action * action_range / 2
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self._steps >= self._max_steps

        info = {}
        self.set_previous_features()
        
        log_dict = self.get_log_dict()
        for key, value in log_dict.items():
            self.current_episode_data[key].append(value)
        
        return observation, reward, terminated, truncated, info
    
    def set_previous_features(self):
        self.last_position = self.data.qpos[[self.qpos_map["body_pos_x"], self.qpos_map["body_pos_y"]]]
        self.last_velocity = self.data.qvel[[self.qvel_map["body_linear_vel_x"], self.qvel_map["body_linear_vel_y"], self.qvel_map["body_linear_vel_z"]]]
        self.last_angular_velocity = self.data.qvel[[self.qvel_map["body_angular_vel_x"], self.qvel_map["body_angular_vel_y"], self.qvel_map["body_angular_vel_z"]]]
        self.last_steer_position = self.data.qpos[[self.qpos_map["steer_FL"], self.qpos_map["steer_FR"], self.qpos_map["steer_BL"], self.qpos_map["steer_BR"]]]
        self.last_steer_velocity = self.data.qvel[[self.qvel_map["steer_FL"], self.qvel_map["steer_FR"], self.qvel_map["steer_BL"], self.qvel_map["steer_BR"]]]
        self.last_drive_velocity = self.data.qvel[[self.qvel_map["drive_FL"], self.qvel_map["drive_FR"], self.qvel_map["drive_BL"], self.qvel_map["drive_BR"]]]
    

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        if self.config.unit_ranges:
            self.action_space = spaces.Box(low=-np.ones_like(low), high=np.ones_like(high), dtype=np.float32)
            self.action_space_raw = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self.action_space_raw = self.action_space
        return self.action_space

    def get_action_space(self):
        return self.action_space
    


    def do_simulation(self, ctrl, n_frames) -> None:
        """
        Step the simulation n number of frames and applying a control action.
        """
        # Check control input is contained in the action space
        if np.array(ctrl).shape != (self.model.nu,):
            raise ValueError(
                f"Action dimension mismatch. Expected {(self.model.nu,)}, found {np.array(ctrl).shape}"
            )
        self._step_mujoco_simulation(ctrl, n_frames)

    def _get_obs(self):
        self.latest_sensor_raw = np.concatenate([self.data.sensordata.copy()])
        steer_pos = self.data.qpos[[self.qpos_map["steer_FL"],
                                    self.qpos_map["steer_FR"],
                                    self.qpos_map["steer_BL"],
                                    self.qpos_map["steer_BR"]]]
        steer_vector_cos = np.cos(steer_pos)
        steer_vector_sin = np.sin(steer_pos)
        
        self.latest_observation = np.concatenate([
            steer_vector_cos.flatten(), # cos of wheel steer positions (4)
            steer_vector_sin.flatten(), # sin of wheel steer positions (4)
            self.latest_sensor_raw[[self.sensor_map["steervel_FL"], # wheel steer velocities (4)
                                    self.sensor_map["steervel_FR"],
                                    self.sensor_map["steervel_BL"],
                                    self.sensor_map["steervel_BR"]]],
            self.latest_sensor_raw[[self.sensor_map["drivevel_FL"], # wheel drive velocities (4)
                                    self.sensor_map["drivevel_FR"],
                                    self.sensor_map["drivevel_BL"],
                                    self.sensor_map["drivevel_BR"]]],
            self.latest_sensor_raw[[self.sensor_map["body_linvel_x"], # body linear velocity (3)
                                    self.sensor_map["body_linvel_y"],
                                    self.sensor_map["body_linvel_z"]]],
            self.latest_sensor_raw[[self.sensor_map["body_angvel_x"], # body angular velocity (3)
                                    self.sensor_map["body_angvel_y"],
                                    self.sensor_map["body_angvel_z"]]],
            self.latest_sensor_raw[[self.sensor_map["body_xaxis_1"], # body x-axis orientation (3)
                                    self.sensor_map["body_xaxis_2"],
                                    self.sensor_map["body_xaxis_3"]]],
            self.latest_sensor_raw[[self.sensor_map["body_pos_x"], # body position (2)
                                    self.sensor_map["body_pos_y"]]]
        ])

        if self.config.unit_ranges:
            observation_max = self.observation_space_raw.high
            observation_min = self.observation_space_raw.low
            observation_center = (observation_max + observation_min) / 2
            observation_range = observation_max - observation_min
            self.latest_observation = (self.latest_observation - observation_center) / observation_range
        return self.latest_observation


    def reset_model(self):
        self._steps = 0
        self.update_episode_data()
        self.current_episode_data.clear()
        
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        #qpos
        reset_angle = self.np_random.uniform(0, 2 * np.pi)
        reset_direction = np.array([np.cos(reset_angle), np.sin(reset_angle)])
        reset_distance = self.np_random.uniform(low=1, high=self.config.reset_pos_offset)*0.1*self.config.reset_noise_scale
        reset_pos = reset_direction * reset_distance +np.array([-self.config.target_pos_distance ,-self.config.target_pos_distance])
        qpos[[self.qpos_map["body_pos_x"], self.qpos_map["body_pos_y"]]] = reset_pos
        qpos[self.qpos_map["body_pos_z"]] += 0
        # Generate a random rotation around the z-axis
        angle_center = np.pi/4
        angle_range = np.radians(30)*self.config.reset_noise_scale
        angle = self.np_random.uniform(angle_center - angle_range, angle_center + angle_range)
        qpos[self.qpos_map["body_quaternion_x"]] = np.sin(angle / 2) * 0
        qpos[self.qpos_map["body_quaternion_y"]] = np.sin(angle / 2) * 0
        qpos[self.qpos_map["body_quaternion_z"]] = np.sin(angle / 2) * 1
        qpos[self.qpos_map["body_quaternion_w"]] = np.cos(angle / 2)
        qpos[[self.qpos_map["drive_BL"], self.qpos_map["drive_BR"], 
              self.qpos_map["drive_FL"], self.qpos_map["drive_FR"]]] += 0
        qpos[[self.qpos_map["slide_BL"], self.qpos_map["slide_BR"],
              self.qpos_map["slide_FL"], self.qpos_map["slide_FR"]]] += 0
        qpos[[self.qpos_map["steer_BL"], self.qpos_map["steer_BR"],
              self.qpos_map["steer_FL"], self.qpos_map["steer_FR"]]] += 0
        # qvel
        qvel[self.qvel_map["body_linear_vel_x"]] += 0
        qvel[self.qvel_map["body_linear_vel_y"]] += 0
        qvel[self.qvel_map["body_linear_vel_z"]] += 0
        qvel[self.qvel_map["body_angular_vel_x"]] += 0
        qvel[self.qvel_map["body_angular_vel_y"]] += 0
        qvel[self.qvel_map["body_angular_vel_z"]] += 0
        qvel[[self.qvel_map["slide_FL"], self.qvel_map["slide_FR"],
              self.qvel_map["slide_BL"], self.qvel_map["slide_BR"]]] += 0
        qvel[[self.qvel_map["steer_FL"], self.qvel_map["steer_FR"],
              self.qvel_map["steer_BL"], self.qvel_map["steer_BR"]]] += 0
        qvel[[self.qvel_map["drive_FL"], self.qvel_map["drive_FR"],
              self.qvel_map["drive_BL"], self.qvel_map["drive_BR"]]] += 0

        self.set_state(qpos, qvel)
        self.get_initial_distance(reset=True)
        self.set_previous_features()
        return self._get_obs()
    
    def get_initial_distance(self,reset=False):
        if reset:
            self.initial_distance = np.linalg.norm(self.target_pos - self.data.qpos[[self.qpos_map["body_pos_x"], self.qpos_map["body_pos_y"]]])
        return self.initial_distance
    
    def _compute_reward(self):
        self.reward_breakdown = {}
        # Position error
        position = self.latest_sensor_raw[[self.sensor_map["body_pos_x"],
                                           self.sensor_map["body_pos_y"]]]
        position_error = np.linalg.norm(position - self.target_pos)
        # Distance closed to target
        distance_closed_to_target = self.get_initial_distance() - position_error
        # Position error change
        last_position_error = np.linalg.norm(self.last_position - self.target_pos)
        position_error_change = last_position_error - position_error
        position_error_velocity = position_error_change / self.dt
        #### Rewards ####
        
        position_error_change_scaled = position_error_change * self.config.reward_scale_position_error_change
        self.reward_breakdown["position_error_change"] = position_error_change
        self.reward_breakdown["position_error_change_scaled"] = position_error_change_scaled
        
        position_error_velocity_scaled = position_error_velocity * self.config.reward_scale_position_error_velocity
        self.reward_breakdown["position_error_velocity"] = position_error_velocity
        self.reward_breakdown["position_error_velocity_scaled"] = position_error_velocity_scaled
        
        distance_closed_to_target_scaled = distance_closed_to_target * self.config.reward_scale_distance_closed_to_target
        self.reward_breakdown["distance_closed_to_target"] = distance_closed_to_target
        self.reward_breakdown["distance_closed_to_target_scaled"] = distance_closed_to_target_scaled
        
        success_reward = 0 if position_error > 0.5 else 1
        success_reward_scaled = success_reward * self.config.reward_scale_success_reward
        self.reward_breakdown["success_reward"] = success_reward
        self.reward_breakdown["success_reward_scaled"] = success_reward_scaled
        #### Penalties ####
        position_penalty = -position_error
        position_penalty_scaled = position_penalty * self.config.reward_scale_position_penalty
        self.reward_breakdown["position_penalty"] = position_penalty
        self.reward_breakdown["position_penalty_scaled"] = position_penalty_scaled
        # Wheel alignment penalty
        wheel_vectors = np.array([ # Transpose to get 4 vectors of shape (2,)
            np.cos(self.data.sensordata[[self.sensor_map["steerpos_FL"],
                                          self.sensor_map["steerpos_FR"],
                                          self.sensor_map["steerpos_BL"],
                                          self.sensor_map["steerpos_BR"]]]),
            np.sin(self.data.sensordata[[self.sensor_map["steerpos_FL"],
                                          self.sensor_map["steerpos_FR"],
                                          self.sensor_map["steerpos_BL"],
                                          self.sensor_map["steerpos_BR"]]])
        ]).T 
        dot_products = [ # Calculate dot products between adjacent wheels
            np.dot(wheel_vectors[i], wheel_vectors[(i+1)%4])
            for i in range(4)
        ]
        # Penalize misalignment, but allow some deviation
        target_alignment = np.cos(np.radians(self.config.wheel_alignment_threshold))  # Allow 10 degrees of misalignment
        misalignment_penalties = [
            max(0, target_alignment - abs(dot))
            for dot in dot_products
        ]
        wheel_misalignment_penalty = - np.sum(misalignment_penalties)
        wheel_misalignment_penalty_scaled = wheel_misalignment_penalty * self.config.reward_scale_wheel_misalignment
        self.reward_breakdown["wheel_misalignment_penalty"] = wheel_misalignment_penalty
        self.reward_breakdown["wheel_misalignment_penalty_scaled"] = wheel_misalignment_penalty_scaled
        
        # Action penalty
        drive_action = self.data.ctrl[[self.ctrl_map["motor_FL_drive"],
                                      self.ctrl_map["motor_FR_drive"],
                                      self.ctrl_map["motor_BL_drive"],
                                      self.ctrl_map["motor_BR_drive"]]]
        steer_action = self.data.ctrl[[self.ctrl_map["motor_FL_steer"],
                                      self.ctrl_map["motor_FR_steer"],
                                      self.ctrl_map["motor_BL_steer"],
                                      self.ctrl_map["motor_BR_steer"]]]
        

        steer_action_penalty = -np.sum(np.square(steer_action))
        drive_action_penalty = -np.sum(np.square(drive_action))

        
        
        scaled_drive_action_penalty = drive_action_penalty * self.config.reward_scale_drive_action_penalty
        scaled_steer_action_penalty = steer_action_penalty * self.config.reward_scale_steer_action_penalty
        self.reward_breakdown["steer_action_penalty"] = steer_action_penalty
        self.reward_breakdown["drive_action_penalty"] = drive_action_penalty
        self.reward_breakdown["drive_action_penalty_scaled"] = scaled_drive_action_penalty
        self.reward_breakdown["steer_action_penalty_scaled"] = scaled_steer_action_penalty

        action_penalty = steer_action_penalty + drive_action_penalty
        action_penalty_scaled = (scaled_drive_action_penalty + scaled_steer_action_penalty) * self.config.reward_scale_action_penalty
        self.reward_breakdown["action_penalty"] = action_penalty
        self.reward_breakdown["action_penalty_scaled"] = action_penalty_scaled

        total_reward = success_reward_scaled + distance_closed_to_target_scaled + position_error_change_scaled + position_error_velocity_scaled
        total_penalty = position_penalty_scaled + wheel_misalignment_penalty_scaled + action_penalty_scaled
        total_reward_scaled = self.dt * (total_reward + total_penalty) * self.config.reward_scale_total
        self.reward_breakdown["total_reward"] = total_reward
        self.reward_breakdown["total_penalty"] = total_penalty
        self.reward_breakdown["total_reward_scaled"] = total_reward_scaled
        return total_reward_scaled
    
    def _check_termination(self):
        return not np.isfinite(self._get_obs()).all()
    
    def get_state(self):
        return self.state_vector()

    def get_log_dict(self):
        log_dict = {}
        # qpos using mappings
        if self.config.log_state:   
            for key, value in self.qpos_map.items():
                log_dict[f"state/{key}"] = self.data.qpos[value]
        # qvel using mappings
        if self.config.log_state:
            for key, value in self.qvel_map.items():
                log_dict[f"state/{key}"] = self.data.qvel[value]   
        # sensor data using mappings
        if self.config.log_sensor:
            for key, value in self.sensor_map.items():
                log_dict[f"sensor/{key}"] = self.latest_sensor_raw[value]
        # ctrl using mappings
        if self.config.log_ctrl:    
            for key, value in self.ctrl_map.items():
                log_dict[f"ctrl/{key}"] = self.data.ctrl[value]
        # reward breakdown  
        if self.config.log_reward:
            for key, value in self.reward_breakdown.items():
                log_dict[f"reward/{key}"] = value
        # observation breakdown
        if self.config.log_observation:
            for key, value in self.observation_map.items():
                log_dict[f"observation/{key}"] = self.latest_observation[value]
        return log_dict

    def update_episode_data(self):
        if self.current_episode_data:
            self.episode_count += 1
            self.previous_episode_data = self.current_episode_data.copy()

    def get_current_episode_data(self):
        return dict(self.current_episode_data)

    def get_previous_episode_data(self):
        return dict(self.previous_episode_data)

    def plot_summary(self,file_path=None,episode_data=None):
        if episode_data is None:
            episode_data = self.current_episode_data if len(self.current_episode_data['sensor/body_pos_x']) > 0 else self.previous_episode_data
        # Check if there's enough data to plot
        if not episode_data or len(episode_data['sensor/body_pos_x']) == 0:
            print("Not enough data to generate a plot. Try running the simulation longer.")
            return None

        # Extract position and velocity data
        x_positions = episode_data['sensor/body_pos_x']
        y_positions = episode_data['sensor/body_pos_y']
        x_velocities = episode_data['sensor/body_linvel_x']
        y_velocities = episode_data['sensor/body_linvel_y']

        # Calculate speed
        speeds = np.sqrt(np.array(x_velocities)**2 + np.array(y_velocities)**2)

        # Create a custom colormap
        colors = ['blue', 'green', 'yellow', 'red']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the trajectory
        scatter = ax.scatter(x_positions, y_positions, c=speeds, cmap=cmap, s=10)

        # Plot start and end points
        ax.plot(x_positions[0], y_positions[0], 'go', markersize=15, label='Start')
        ax.plot(x_positions[-1], y_positions[-1], 'ro', markersize=15, label='End')

        # Plot target position
        ax.plot(self.target_pos[0], self.target_pos[1], 'y*', markersize=50, label='Target')

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Speed')

        # Set labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Quadbot Trajectory')
        ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Ensure equal aspect ratio
        ax.set_aspect('equal', 'box')
        
        if not file_path:
            file_path = f"trajectory/quadbot_trajectory_{time.time()}.jpeg"
        # check if dir exists
        if not os.path.exists("trajectory"):
            os.makedirs("trajectory")
        plt.savefig(file_path, format='jpeg',dpi=150, bbox_inches='tight',pil_kwargs={'quality': 10})
        # Close the plot to free up memory
        plt.close(fig)

        return file_path

    def get_episode_summary(self, episode_data):
        summary = {}
        for key, values in episode_data.items():
            summary[key] = {
                'mean': np.mean(values),
                'max': np.max(values),
                'min': np.min(values),
                'median': np.median(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75)
            }
        

        return summary

    def get_current_episode_summary(self):
        return self.get_episode_summary(self.current_episode_data)

    def get_previous_episode_summary(self):
        return self.get_episode_summary(self.previous_episode_data)

    def episode_log_summary(self):
        return {
            'current_episode': self.get_current_episode_summary(),
            'previous_episode': self.get_previous_episode_summary()
        }

if __name__ == "__main__":
    import MyUtils.Util.Misc as mutil
    # Change dir to current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if os.path.exists("base_config.yaml"):
        cfg = mutil.load_and_override_config(".", "base_config", init_wandb=False, update_wandb=False)
        cfg.env.log_state = True
        cfg.env.log_sensor = True
        cfg.env.log_reward = True
        cfg.env.log_observation = True
        env = QuadChaseEnv(render_mode="human", max_steps=10000, config=cfg.env)
    else:
        env = QuadChaseEnv(render_mode="human", max_steps=10000)
    util.print_joint_info(env.model)

    observation, info = env.reset()
    terminated = truncated = False
    total_reward = 0

    drive_action = 0
    steer_action = 0
    running = True
    reset_flag = False
    plot_flag = False  # New flag for plotting

    print("Use arrow keys to control the quadbot:")
    print("Up/Down: Control drive motors")
    print("Left/Right: Control steering")
    print("Press 'q' to quit")
    print("Press 'p' to generate and save the plot")  # New instruction

    def on_press(key):
        global drive_action, steer_action, running, reset_flag, plot_flag
        try:
            if key == keyboard.Key.up:
                drive_action = 1    
            elif key == keyboard.Key.down:
                drive_action = -1
            elif key == keyboard.Key.left:
                steer_action = -0.5
            elif key == keyboard.Key.right:
                steer_action = 0.5
            elif key == keyboard.KeyCode.from_char('q'):
                running = False
            elif key == keyboard.KeyCode.from_char('å'):
                reset_flag = True
            elif key == keyboard.KeyCode.from_char('p'):  # New key for plotting
                plot_flag = True
        except AttributeError:
            pass

    def on_release(key):
        global drive_action, steer_action
        if key in [keyboard.Key.up, keyboard.Key.down]:
            drive_action = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            steer_action = 0

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        last_step_time = time.time()
        while running:  
            if reset_flag:
                observation, info = env.reset()
                reset_flag = False
            elif plot_flag:
                print("Generating plot...")
                plot_buffer = env.plot_summary()
                if plot_buffer is not None:
                    with open("quadbot_trajectory.png", "wb") as f:
                        f.write(plot_buffer.getbuffer())
                    print("Plot saved as 'quadbot_trajectory.png'")
                plot_flag = False
            else:
                action = [drive_action] * 4 + [steer_action] * 4
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                state = env.get_state()
                reward_dict = env.get_log_dict()
                for k, v in reward_dict.items():
                    if k.startswith("reward/"):
                        print(f"{k}: {v:.2f}")
                current_time = time.time()
                step_duration = current_time - last_step_time
                steps_per_second = 1 / step_duration if step_duration > 0 else 0
                last_step_time = current_time
                print(f"Reward: {reward/env.dt:.2f} action: {action}, observation: {observation}, Steps/s: {steps_per_second:.2f}")
                env.render()
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print(f"Total Reward: {total_reward}")
        env.close()
        listener.stop()
