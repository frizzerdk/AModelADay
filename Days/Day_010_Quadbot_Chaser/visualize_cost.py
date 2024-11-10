import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import os

class RewardCalculator:
    def __init__(self):
        # Default reward scaling parameters
        self.config = {
            'reward_scale_action_penalty': 1.0,
            'reward_scale_drive_action_penalty': 0.001,
            'reward_scale_steer_action_penalty': 1.0,
            'reward_scale_wheel_misalignment': 0.5,
            'reward_scale_position_penalty': 0.0,
            'reward_scale_success_reward': 50.0,
            'reward_scale_position_error_change': 0.0,
            'reward_scale_distance_closed_to_target': 0.0,
            'reward_scale_position_error_velocity': 50.0,
            'reward_scale_total': 10.0,
            'wheel_alignment_threshold': 10,
        }
        self.dt = 0.002  # Simulation timestep

    def compute_reward_components(self, pos, vel, target_pos, wheel_angles, initial_distance=None):
        reward_breakdown = {}
        
        # Position error
        position_error = np.linalg.norm(pos - target_pos)
        
        # Distance closed to target
        if initial_distance is None:
            initial_distance = position_error
        distance_closed_to_target = initial_distance - position_error
        
        # Position error velocity
        position_error_velocity = np.dot(vel, (target_pos - pos)) / (position_error + 1e-6)

        # Wheel alignment penalty
        wheel_vectors = np.array([
            [np.cos(angle), np.sin(angle)] for angle in wheel_angles
        ])
        dot_products = [
            np.dot(wheel_vectors[i], wheel_vectors[(i+1)%4])
            for i in range(4)
        ]
        target_alignment = np.cos(np.radians(self.config['wheel_alignment_threshold']))
        misalignment_penalties = [
            max(0, target_alignment - abs(dot))
            for dot in dot_products
        ]
        wheel_misalignment_penalty = -np.sum(misalignment_penalties)
        wheel_misalignment_penalty_scaled = wheel_misalignment_penalty * self.config['reward_scale_wheel_misalignment']
        reward_breakdown["wheel_misalignment_penalty"] = wheel_misalignment_penalty
        reward_breakdown["wheel_misalignment_penalty_scaled"] = wheel_misalignment_penalty_scaled

        #### Rewards ####
        # Success reward
        success_reward = 1.0 if position_error < 0.5 else 0.0
        success_reward_scaled = success_reward * self.config['reward_scale_success_reward']
        reward_breakdown["success_reward"] = success_reward
        reward_breakdown["success_reward_scaled"] = success_reward_scaled
        
        # Position error velocity reward
        position_error_velocity_scaled = position_error_velocity * self.config['reward_scale_position_error_velocity']
        reward_breakdown["position_error_velocity"] = position_error_velocity
        reward_breakdown["position_error_velocity_scaled"] = position_error_velocity_scaled
        
        # Distance closed reward
        distance_closed_scaled = distance_closed_to_target * self.config['reward_scale_distance_closed_to_target']
        reward_breakdown["distance_closed_to_target"] = distance_closed_to_target
        reward_breakdown["distance_closed_to_target_scaled"] = distance_closed_scaled

        #### Penalties ####
        # Position penalty
        position_penalty = -position_error
        position_penalty_scaled = position_penalty * self.config['reward_scale_position_penalty']
        reward_breakdown["position_penalty"] = position_penalty
        reward_breakdown["position_penalty_scaled"] = position_penalty_scaled

        # Calculate total reward
        total_reward = (success_reward_scaled + 
                       position_error_velocity_scaled + 
                       distance_closed_scaled + 
                       position_penalty_scaled +
                       wheel_misalignment_penalty_scaled)
        
        total_reward_scaled = self.dt * total_reward * self.config['reward_scale_total']
        reward_breakdown["total_reward"] = total_reward
        reward_breakdown["total_reward_scaled"] = total_reward_scaled
        
        return reward_breakdown

class CostVisualizer:
    def __init__(self):
        self.reward_calculator = RewardCalculator()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Cost Function Visualizer")
        self.root.geometry("1400x800")

        # Create frames
        left_frame = ttk.Frame(self.root, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        middle_frame = ttk.Frame(self.root, padding="5")
        middle_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        right_frame = ttk.Frame(self.root, padding="5")
        right_frame.grid(row=0, column=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create position and velocity sliders
        self.pos_x = self.create_slider(left_frame, "Position X", -5, 5, 0, 0)
        self.pos_y = self.create_slider(left_frame, "Position Y", -5, 5, 0, 1)
        self.vel_x = self.create_slider(left_frame, "Velocity X", -2, 2, 0, 2)
        self.vel_y = self.create_slider(left_frame, "Velocity Y", -2, 2, 0, 3)
        self.target_x = self.create_slider(left_frame, "Target X", -5, 5, 0, 4)
        self.target_y = self.create_slider(left_frame, "Target Y", -5, 5, 3, 5)

        # Create wheel angle sliders
        ttk.Label(middle_frame, text="Wheel Angles (radians)").grid(row=0, column=0, columnspan=2)
        self.wheel_FL = self.create_slider(middle_frame, "Front Left", -np.pi, np.pi, 0, 1)
        self.wheel_FR = self.create_slider(middle_frame, "Front Right", -np.pi, np.pi, 0, 2)
        self.wheel_BL = self.create_slider(middle_frame, "Back Left", -np.pi, np.pi, 0, 3)
        self.wheel_BR = self.create_slider(middle_frame, "Back Right", -np.pi, np.pi, 0, 4)

        # Create reward scaling sliders
        row = 0
        self.reward_sliders = {}
        for key, value in self.reward_calculator.config.items():
            if key.startswith('reward_scale'):
                self.reward_sliders[key] = self.create_slider(
                    right_frame, key, 0, value * 2, value, row)
                row += 1

        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        # Bind slider updates
        all_sliders = [
            self.pos_x, self.pos_y, self.vel_x, self.vel_y, 
            self.target_x, self.target_y,
            self.wheel_FL, self.wheel_FR, self.wheel_BL, self.wheel_BR,
            *self.reward_sliders.values()
        ]
        for slider in all_sliders:
            slider.configure(command=lambda _: self.update_plot())

        # Initial plot
        self.update_plot()

    def create_slider(self, parent, label, min_val, max_val, default, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=5, pady=5)
        slider = ttk.Scale(parent, from_=min_val, to=max_val, orient=tk.HORIZONTAL)
        slider.set(default)
        slider.grid(row=row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        return slider

    def update_plot(self):
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Get values from sliders
        pos = np.array([self.pos_x.get(), self.pos_y.get()])
        vel = np.array([self.vel_x.get(), self.vel_y.get()])
        target_pos = np.array([self.target_x.get(), self.target_y.get()])
        wheel_angles = np.array([
            self.wheel_FL.get(),
            self.wheel_FR.get(),
            self.wheel_BL.get(),
            self.wheel_BR.get()
        ])

        # Update reward scales
        for key, slider in self.reward_sliders.items():
            self.reward_calculator.config[key] = slider.get()

        # Calculate reward components
        reward_info = self.reward_calculator.compute_reward_components(pos, vel, target_pos, wheel_angles)

        # Plot 1: Position visualization
        self.ax1.scatter(pos[0], pos[1], color='blue', s=100, label='Robot')
        self.ax1.scatter(target_pos[0], target_pos[1], color='red', s=100, label='Target')
        self.ax1.quiver(pos[0], pos[1], vel[0], vel[1], color='green', scale=20, label='Velocity')
        self.ax1.set_xlim(-5, 5)
        self.ax1.set_ylim(-5, 5)
        self.ax1.grid(True)
        self.ax1.legend()
        self.ax1.set_title('Position and Velocity')

        # Plot 2: Wheel orientations
        wheel_positions = np.array([
            [0.5, 0.5],   # FL
            [-0.5, 0.5],  # FR
            [0.5, -0.5],  # BL
            [-0.5, -0.5]  # BR
        ])
        wheel_vectors = np.array([
            [np.cos(angle), np.sin(angle)] for angle in wheel_angles
        ]) * 0.3  # Scale the vectors

        self.ax2.set_xlim(-1, 1)
        self.ax2.set_ylim(-1, 1)
        for pos, vec in zip(wheel_positions, wheel_vectors):
            self.ax2.quiver(pos[0], pos[1], vec[0], vec[1], 
                          angles='xy', scale_units='xy', scale=1)
        self.ax2.set_title('Wheel Orientations')
        self.ax2.grid(True)
        self.ax2.set_aspect('equal')

        # Plot 3: Reward components
        components = list(reward_info.keys())
        values = list(reward_info.values())
        
        self.ax3.bar(components, values)
        self.ax3.set_xticklabels(components, rotation=45, ha='right')
        self.ax3.set_title('Reward Components')

        plt.tight_layout()
        self.canvas.draw()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    visualizer = CostVisualizer()
    visualizer.run() 