import numpy as np
from gymnasium import utils
from mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import os
import mujoco
from pynput import keyboard
import time

# State Space Description:
#
# 1. Free Joint (body_joint):
#    qpos[0:3]: Body position (x, y, z)
#               Corresponds to sensor: framepos (body_pos)
#    qpos[3:7]: Body orientation (quaternion)
#               Related to sensor: framexaxis (body_xaxis)
#    qvel[0:3]: Body linear velocity
#               Corresponds to sensor: framelinvel (body_linvel)
#    qvel[3:6]: Body angular velocity
#               No direct sensor
#
# 2. Slide Joints:
#    qpos[7]: slide_FL_joint position
#    qpos[10]: slide_FR_joint position
#    qpos[13]: slide_BL_joint position
#    qpos[16]: slide_BR_joint position
#    qvel[6]: slide_FL_joint velocity
#    qvel[9]: slide_FR_joint velocity
#    qvel[12]: slide_BL_joint velocity
#    qvel[15]: slide_BR_joint velocity
#    Note: These slide joints don't have direct sensors
#
# 3. Steer Joints:
#    qpos[8]: steer_FL_joint angle
#             Corresponds to sensor: jointpos (steerpos_FL)
#    qpos[11]: steer_FR_joint angle
#              Corresponds to sensor: jointpos (steerpos_FR)
#    qpos[14]: steer_BL_joint angle
#              Corresponds to sensor: jointpos (steerpos_BL)
#    qpos[17]: steer_BR_joint angle
#              Corresponds to sensor: jointpos (steerpos_BR)
#    qvel[7]: steer_FL_joint angular velocity
#    qvel[10]: steer_FR_joint angular velocity
#    qvel[13]: steer_BL_joint angular velocity
#    qvel[16]: steer_BR_joint angular velocity
#    Note: The steer joint velocities don't have direct sensors
#
# 4. Drive Joints:
#    qpos[9]: drive_FL_joint angle
#    qpos[12]: drive_FR_joint angle
#    qpos[15]: drive_BL_joint angle
#    qpos[18]: drive_BR_joint angle
#    qvel[8]: drive_FL_joint angular velocity
#             Corresponds to sensor: jointvel (drivevel_FL)
#    qvel[11]: drive_FR_joint angular velocity
#              Corresponds to sensor: jointvel (drivevel_FR)
#    qvel[14]: drive_BL_joint angular velocity
#              Corresponds to sensor: jointvel (drivevel_BL)
#    qvel[17]: drive_BR_joint angular velocity
#              Corresponds to sensor: jointvel (drivevel_BR)
#
# Actuators:
#    ctrl[0]: motor_FL_drive
#    ctrl[1]: motor_FR_drive
#    ctrl[2]: motor_BL_drive
#    ctrl[3]: motor_BR_drive
#    ctrl[4]: motor_FL_steer
#    ctrl[5]: motor_FR_steer
#    ctrl[6]: motor_BL_steer
#    ctrl[7]: motor_BR_steer
# Sensors:
#    sensor[0]: drivevel_FL
#    sensor[1]: drivevel_FR
#    sensor[2]: drivevel_BL
#    sensor[3]: drivevel_BR
#    sensor[4]: steerpos_FL
#    sensor[5]: steerpos_FR
#    sensor[6]: steerpos_BL
#    sensor[7]: steerpos_BR
#    sensor[8]: bodypos_x
#    sensor[9]: bodypos_y
#    sensor[10]: bodypos_z
#    sensor[11]: bodyvel_x
#    sensor[12]: bodyvel_y
#    sensor[13]: bodyvel_z
#    sensor[14]: body_xaxis_1
#    sensor[15]: body_xaxis_2
#    sensor[16]: body_xaxis_3
DEFAULT_CAMERA_CONFIG = {
    "distance": 15,
}

class QuadbotEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "quadbot.xml",
        frame_skip: int = 5,
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.1,
        max_steps: int = 500,
        **kwargs
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, max_steps, **kwargs)
        self.state_dict = {}
        self._reset_noise_scale = reset_noise_scale
        self._max_steps = max_steps
        self._steps = 0
        self.reward_dict = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, xml_file)

        # Define min and max values for each observation component
        obs_low = np.array([
            -1, -1, -1, -1,  # cos of wheel steer positions (4)
            -1, -1, -1, -1,  # sin of wheel steer positions (4)
            -50, -50, -50, -50,  # wheel steer velocities (4)
            -1500, -1500, -1500, -1500,  # wheel drive velocities (4)
            -20, -20, -20,  # body linear velocity (3)
            -1, -1, -1  # body x-axis orientation (3)
        ])

        obs_high = np.array([
            1, 1, 1, 1,  # cos of wheel steer positions (4)
            1, 1, 1, 1,  # sin of wheel steer positions (4)
            50, 50, 50, 50,  # wheel steer velocities (4)
            1500, 1500, 1500, 1500,  # wheel drive velocities (4)
            20, 20, 20,  # body linear velocity (3)
            1, 1, 1  # body x-axis orientation (3)
        ])

        observation_space = Box(low=obs_low, high=obs_high, dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs
        )

        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        self.prev_steer_pos = np.zeros(4)  # Add this line to store previous steer positions
        self.steer_velocities = np.zeros(4)  # Add this line to store steer velocities

    def step(self, action):
        self._steps += 1  # Add this line
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self._steps >= self._max_steps  # Add this line
        info = {
            "reward_forward": reward,
            "reward_breakdown": self.reward_dict  # Add this line to include the reward breakdown in the info dict
        }

        if self.render_mode == "human":
            self.render()
        self.prev_steer_pos = self.raw_latest_observation[0:4]
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self._steps = 0
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Define noise scales for different components
        body_pos_noise_scale = np.array([5., 5., 0.]) 
        body_orientation_noise_scale = 0.01 
        slide_joint_noise_scale = 0.001 
        steer_joint_noise_scale = 0.001 
        drive_joint_noise_scale = 0.001 
        velocity_noise_scale = 0.001 

        # Body position (x, y, z)
        qpos[0:3] += body_pos_noise_scale*self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=3)
        
        # Body orientation (quaternion)
        qpos[3:7] += body_orientation_noise_scale*self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=4)
        qpos[3:7] /= np.linalg.norm(qpos[3:7])  # Normalize quaternion
        
        # Slide joints
        qpos[[7, 10, 13, 16]] += slide_joint_noise_scale*self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=4)
        
        # Steer joints
        qpos[[8, 11, 14, 17]] += steer_joint_noise_scale*self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=4)
        
        # Drive joints
        qpos[[9, 12, 15, 18]] += drive_joint_noise_scale*self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=4)
        
        # Velocities (including body, slide, steer, and drive joint velocities)
        qvel[:] += velocity_noise_scale*self.np_random.uniform(low=-self._reset_noise_scale, high=self._reset_noise_scale, size=qvel.shape)

        self.set_state(qpos, qvel)

        # Ensure the quadbot is not intersecting with the ground
        while self.data.qpos[2] < 0.2:  # Adjust this value based on your model
            qpos[2] += 0.01
            self.set_state(qpos, qvel)

        # Reset previous steer positions
        self.prev_steer_pos = self.data.sensordata[0:4]

        return self._get_obs()
    
    def get_state(self):
        # Existing sensor data
        for i, name in enumerate(['steerpos_FL', 'steerpos_FR', 'steerpos_BL', 'steerpos_BR',
                                  'drivevel_FL', 'drivevel_FR', 'drivevel_BL', 'drivevel_BR',
                                  'bodyvel_x', 'bodyvel_y', 'bodyvel_z',
                                  'body_xaxis_1', 'body_xaxis_2', 'body_xaxis_3']):
            self.state_dict[name] = self.data.sensordata[i]

        # All qpos values
        for i in range(len(self.data.qpos)):
            self.state_dict[f"qpos[{i}]"] = self.data.qpos[i]

        # All qvel values
        for i in range(len(self.data.qvel)):
            self.state_dict[f"qvel[{i}]"] = self.data.qvel[i]

        # Control input (actuator values)
        for i in range(len(self.data.ctrl)):
            self.state_dict[f"ctrl[{i}]"] = self.data.ctrl[i]

        # Add steer velocities

        self.state_dict['steervel_FL'] = self.steer_velocities[0]
        self.state_dict['steervel_FR'] = self.steer_velocities[1]
        self.state_dict['steervel_BL'] = self.steer_velocities[2]
        self.state_dict['steervel_BR'] = self.steer_velocities[3]

        return self.state_dict

    def _get_obs(self):
        # Observation state explanation:
        #    [0-3] Wheel steer positions (FL, FR, BL, BR)
        #    [4-7] Wheel drive velocities (FL, FR, BL, BR)
        #    [8-11] Wheel steer velocities (FL, FR, BL, BR)
        #    [12-14] Body linear velocity (x, y, z)
        #    [15-17] Body x-axis orientation
        # Total: 18 observation values
        self.raw_latest_observation = np.concatenate([self.data.sensordata.copy()])
        
        # Calculate steer velocities
        current_steer_pos = self.raw_latest_observation[0:4]
        self.steer_velocities = (current_steer_pos - self.prev_steer_pos) / self.dt
        
        
        self.latest_observation = np.concatenate([
            np.cos(current_steer_pos), # Wheel steer positions (4)
            np.sin(current_steer_pos), # Wheel steer positions (4)
            self.steer_velocities,  # Wheel steer velocities
            self.raw_latest_observation[4:8],  # Wheel drive velocities
            self.raw_latest_observation[8:14]  # Body linear velocity and x-axis orientation
        ])
        
        self.latest_state = self.data.qpos.copy()

        return self.latest_observation

    def _compute_reward(self):
        # Forward motion reward (average of wheel velocities)
        x_velocity = self.data.qvel[0]
        y_velocity_penalty = -np.square(0.1*self.data.qvel[1])
        body_rotation_penalty = -0.01 * np.sum(np.square(0.1*self.data.qvel[3:6]))
        steer_penalty = -1 * np.sum(np.square(0.1*self.data.qvel[[7, 10, 13, 16]]))
        drive_penalty = -1 * np.sum(np.square(0.0001*self.data.qvel[[8, 11, 14, 17]]))

        # Actuation input penalty
        actuation_penalty = -1 * np.sum(np.square(0.1*self.data.ctrl))

        # Wheel alignment penalty
        wheel_vectors = np.array([
            np.cos(self.data.sensordata[0:4]),
            np.sin(self.data.sensordata[0:4])
        ]).T  # Transpose to get 4 vectors of shape (2,)

        # Calculate dot products between adjacent wheels
        dot_products = [
            np.dot(wheel_vectors[i], wheel_vectors[(i+1)%4])
            for i in range(4)
        ]

        # Penalize misalignment, but allow some deviation
        target_alignment = np.cos(np.radians(10))  # Allow 10 degrees of misalignment
        misalignment_penalties = [
            max(0, target_alignment - abs(dot))
            for dot in dot_products
        ]

        wheel_misalignment_penalty = np.sum(misalignment_penalties)

        # Update reward dictionary
        self.reward_dict = {
            "x_velocity": x_velocity,
            "y_velocity_penalty": y_velocity_penalty,
            "body_rotation_penalty": body_rotation_penalty,
            "steer_penalty": steer_penalty,
            "drive_penalty": drive_penalty,
            "actuation_penalty": actuation_penalty,
            "wheel_misalignment_penalty": wheel_misalignment_penalty
        }
        #clip penalty to be between 0 and -1
        def sigmoid_clip(x, lower=-10, upper=0):
            return lower + (upper - lower) * (1 / (1 + np.exp(-x)))

        wheel_misalignment_penalty = sigmoid_clip(wheel_misalignment_penalty)
        y_velocity_penalty = sigmoid_clip(y_velocity_penalty)
        actuation_penalty = sigmoid_clip(actuation_penalty)
        drive_penalty = sigmoid_clip(drive_penalty)
        steer_penalty = sigmoid_clip(steer_penalty)
        body_rotation_penalty = sigmoid_clip(body_rotation_penalty)

        reward_scale = 0.01
        secondary_scale = reward_scale*1
        
        # Combine rewards
        reward = reward_scale*x_velocity + secondary_scale*(y_velocity_penalty + actuation_penalty + drive_penalty + wheel_misalignment_penalty + body_rotation_penalty + steer_penalty)

        return reward 

    def _check_termination(self):
        height = self.data.qpos[2]
        return height < 0.01 or not np.isfinite(self._get_obs()).all()

def print_joint_info(model):
    joint_info = get_joint_info(model)
    for joint in joint_info:
        print(f"Joint: {joint['name']}")
        print(f"  Type: {mujoco.mjtJoint(joint['type']).name}")
        print(f"  qpos indices: {joint['qpos_indices']}")
        print(f"  qvel indices: {joint['qvel_indices']}")
        print()

def get_joint_info(model):
    joint_info = []
    qpos_index = 0
    qvel_index = 0

    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_type = model.jnt_type[i]
        
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_range = range(qpos_index, qpos_index + 7)
            qvel_range = range(qvel_index, qvel_index + 6)
            qpos_index += 7
            qvel_index += 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            qpos_range = range(qpos_index, qpos_index + 4)
            qvel_range = range(qvel_index, qvel_index + 3)
            qpos_index += 4
            qvel_index += 3
        elif joint_type in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
            qpos_range = [qpos_index]
            qvel_range = [qvel_index]
            qpos_index += 1
            qvel_index += 1
        else:
            continue  # Skip other joint types

        joint_info.append({
            'name': joint_name,
            'type': joint_type,
            'qpos_indices': qpos_range,
            'qvel_indices': qvel_range
        })

    return joint_info

if __name__ == "__main__":
    env = QuadbotEnv(render_mode="human", max_steps=10000)
    print_joint_info(env.model)

    observation, info = env.reset()
    terminated = truncated = False
    total_reward = 0

    drive_action = 0
    steer_action = 0
    running = True

    print("Use arrow keys to control the quadbot:")
    print("Up/Down: Control drive motors")
    print("Left/Right: Control steering")
    print("Press 'q' to quit")

    def on_press(key):
        global drive_action, steer_action, running
        try:
            if key == keyboard.Key.up:
                drive_action = 1    
            elif key == keyboard.Key.down:
                drive_action = -1
            elif key == keyboard.Key.left:
                steer_action = -1
            elif key == keyboard.Key.right:
                steer_action = 1
            elif key == keyboard.KeyCode.from_char('q'):
                running = False
        except AttributeError:
            pass

    def on_release(key):
        global drive_action, steer_action
        print(key)
        if key in [keyboard.Key.up, keyboard.Key.down]:
            drive_action = 0
        elif key in [keyboard.Key.left, keyboard.Key.right]:
            steer_action = 0

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while running and not (terminated or truncated):
            action = [drive_action] * 4 + [steer_action] * 4
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Reward: {reward} action: {action}, observation: {observation[8:12]}")
            env.render()
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print(f"Total Reward: {total_reward}")
        env.close()
        listener.stop()