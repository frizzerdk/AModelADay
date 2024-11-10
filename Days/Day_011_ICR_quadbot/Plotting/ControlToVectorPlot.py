import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms

def calculate_wheel_vectors(wheel_positions, icr_x, icr_y, target_speed=1.0):
    """Calculate vectors for a single ICR"""
    wheel_positions = np.array(wheel_positions)
    
    # Handle infinite ICR case
    if np.isinf(icr_x) or np.isinf(icr_y):
        angles = np.zeros(len(wheel_positions))
        speeds = np.ones(len(wheel_positions)) * target_speed
        vectors = np.array([
            np.cos(angles),
            np.sin(angles)
        ]).T * speeds[:, np.newaxis]
        return vectors
    
    # Calculate angles to ICR
    angles = np.arctan2(
        wheel_positions[:, 1] - icr_y,
        wheel_positions[:, 0] - icr_x
    ) + np.pi/2  # Add 90 degrees to make wheels tangent to circles
    
    # Calculate radii from ICR to each wheel
    radii = np.sqrt(
        (wheel_positions[:, 0] - icr_x)**2 + 
        (wheel_positions[:, 1] - icr_y)**2
    )
    
    # Normalize speeds
    speeds = radii / np.mean(radii) * target_speed if np.mean(radii) > 0 else np.ones_like(radii) * target_speed
    
    # Calculate vectors
    vectors = np.array([
        np.cos(angles),
        np.sin(angles)
    ]).T * speeds[:, np.newaxis]
    
    return vectors

class DualICRVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual ICR Vector Visualizer")
        
        # Initialize wheel positions
        self.wheel_positions = np.array([
            [-0.5, 0.5],   # Front Left
            [0.5, 0.5],    # Front Right
            [-0.5, -0.5],  # Back Left
            [0.5, -0.5]    # Back Right
        ])
        
        # Wheel dimensions
        self.wheel_width = 0.2
        self.wheel_height = 0.1
        
        # Create main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create sliders for two ICRs
        self.icr1_x_var = tk.DoubleVar(value=1.0)
        self.icr1_y_var = tk.DoubleVar(value=1.0)
        self.icr1_speed_var = tk.DoubleVar(value=1.0)
        
        self.icr2_x_var = tk.DoubleVar(value=-1.0)
        self.icr2_y_var = tk.DoubleVar(value=1.0)
        self.icr2_speed_var = tk.DoubleVar(value=1.0)
        
        # ICR 1 controls
        ttk.Label(control_frame, text="ICR 1 Controls", font='bold').pack()
        ttk.Label(control_frame, text="X:").pack()
        ttk.Scale(control_frame, from_=-5.0, to=5.0, variable=self.icr1_x_var, 
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Y:").pack()
        ttk.Scale(control_frame, from_=-5.0, to=5.0, variable=self.icr1_y_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Speed:").pack()
        ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.icr1_speed_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # ICR 2 controls
        ttk.Label(control_frame, text="ICR 2 Controls", font='bold').pack()
        ttk.Label(control_frame, text="X:").pack()
        ttk.Scale(control_frame, from_=-5.0, to=5.0, variable=self.icr2_x_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Y:").pack()
        ttk.Scale(control_frame, from_=-5.0, to=5.0, variable=self.icr2_y_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Speed:").pack()
        ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.icr2_speed_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Add simulation controls
        sim_frame = ttk.Frame(control_frame)
        sim_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sim_frame, text="Simulation Controls", font='bold').pack()
        
        # Add simulation time slider
        self.sim_time_var = tk.DoubleVar(value=5.0)
        ttk.Label(sim_frame, text="Simulation Time (s):").pack()
        ttk.Scale(sim_frame, from_=1.0, to=10.0, variable=self.sim_time_var,
                 orient=tk.HORIZONTAL,command=self.update_plot).pack(fill=tk.X)
        
        # Add simulation step slider
        self.sim_steps_var = tk.IntVar(value=100)
        ttk.Label(sim_frame, text="Simulation Steps:").pack()
        ttk.Scale(sim_frame, from_=50, to=500, variable=self.sim_steps_var,
                 orient=tk.HORIZONTAL,command=self.update_plot).pack(fill=tk.X)
        
        # Add simulation button
        self.sim_button = ttk.Button(sim_frame, text="Show Trajectory", 
                                   command=self.simulate_trajectory)
        self.sim_button.pack(pady=5)
        
        # Initialize trajectory storage
        self.trajectory_points = None
        
        # Initial plot
        self.update_plot()
    
    def draw_wheel(self, pos, angle, color):
        """Draw a wheel as a rotated rectangle"""
        rect = Rectangle(
            (pos[0] - self.wheel_width/2, pos[1] - self.wheel_height/2),
            self.wheel_width, self.wheel_height,
            facecolor=color, alpha=0.5, edgecolor='black'
        )
        t = transforms.Affine2D().rotate_around(pos[0], pos[1], angle)
        rect.set_transform(t + self.ax.transData)
        return rect
    
    def simulate_trajectory(self):
        """Simulate and plot the robot's trajectory over time"""
        # Get current ICR and speed values
        icr1_x = self.icr1_x_var.get()
        icr1_y = self.icr1_y_var.get()
        icr1_speed = self.icr1_speed_var.get()
        
        icr2_x = self.icr2_x_var.get()
        icr2_y = self.icr2_y_var.get()
        icr2_speed = self.icr2_speed_var.get()
        
        # Get simulation parameters
        sim_time = self.sim_time_var.get()
        sim_steps = self.sim_steps_var.get()
        dt = sim_time / sim_steps
        
        # Initialize trajectory storage
        self.trajectory_points = np.zeros((sim_steps, len(self.wheel_positions), 2))
        current_positions = np.copy(self.wheel_positions)
        
        # Simulate motion
        for i in range(sim_steps):
            # Calculate vectors for current position
            vectors1 = calculate_wheel_vectors(current_positions, icr1_x, icr1_y, icr1_speed)
            vectors2 = calculate_wheel_vectors(current_positions, icr2_x, icr2_y, icr2_speed)
            
            # Combine vectors
            combined_vectors = vectors1 + vectors2
            
            # Store current positions
            self.trajectory_points[i] = current_positions
            
            # Update positions using combined vectors
            current_positions = current_positions + combined_vectors * dt
        
        # Update plot with trajectory
        #self.update_plot()
    
    def update_plot(self, *args):
        self.ax.clear()
        self.simulate_trajectory()
        # Get current values
        icr1_x = self.icr1_x_var.get()
        icr1_y = self.icr1_y_var.get()
        icr1_speed = self.icr1_speed_var.get()
        
        icr2_x = self.icr2_x_var.get()
        icr2_y = self.icr2_y_var.get()
        icr2_speed = self.icr2_speed_var.get()
        
        # Calculate vectors for both ICRs
        vectors1 = calculate_wheel_vectors(self.wheel_positions, icr1_x, icr1_y, icr1_speed)
        vectors2 = calculate_wheel_vectors(self.wheel_positions, icr2_x, icr2_y, icr2_speed)
        
        # Combine vectors
        combined_vectors = vectors1 + vectors2
        
        # Calculate angles for wheel display from combined vectors
        angles = np.arctan2(combined_vectors[:, 1], combined_vectors[:, 0])
        speeds = np.linalg.norm(combined_vectors, axis=1)
        
        # Plot robot frame
        self.ax.plot(self.wheel_positions[[0,1,3,2,0], 0], 
                    self.wheel_positions[[0,1,3,2,0], 1], 
                    'k-', label='Robot Frame')
        
        # Plot wheels and their vectors
        colors = ['blue', 'red', 'green', 'purple']
        labels = ['FL', 'FR', 'BL', 'BR']
        
        for i, (pos, vec1, vec2, combined_vec, angle, speed, color, label) in enumerate(
                zip(self.wheel_positions, vectors1, vectors2, combined_vectors, 
                    angles, speeds, colors, labels)):
            
            # Draw wheel rectangle
            wheel = self.draw_wheel(pos, angle, color)
            self.ax.add_patch(wheel)
            
            # Plot individual vectors (thinner, semi-transparent)
            self.ax.arrow(pos[0], pos[1], 
                         vec1[0]*0.2, vec1[1]*0.2,
                         head_width=0.03, head_length=0.05,
                         fc=color, ec=color, alpha=0.3)
            self.ax.arrow(pos[0], pos[1], 
                         vec2[0]*0.2, vec2[1]*0.2,
                         head_width=0.03, head_length=0.05,
                         fc=color, ec=color, alpha=0.3)
            
            # Plot combined vector (thicker)
            self.ax.arrow(pos[0], pos[1], 
                         combined_vec[0]*0.2, combined_vec[1]*0.2,
                         head_width=0.05, head_length=0.1,
                         fc=color, ec=color,
                         label=f'{label} wheel (speed={speed:.2f})')
        
        # Plot ICR points
        self.ax.plot(icr1_x, icr1_y, 'rx', markersize=10, 
                    label=f'ICR 1 ({icr1_x:.1f}, {icr1_y:.1f})')
        self.ax.plot(icr2_x, icr2_y, 'bx', markersize=10, 
                    label=f'ICR 2 ({icr2_x:.1f}, {icr2_y:.1f})')
        
        # Plot trajectories if they exist
        if self.trajectory_points is not None:
            for wheel_idx in range(len(self.wheel_positions)):
                trajectory = self.trajectory_points[:, wheel_idx]
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], '--', 
                           color=colors[wheel_idx], alpha=0.5,
                           label=f'{labels[wheel_idx]} trajectory')
        
        # Set plot properties
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_title('Combined ICR Vector Fields')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Set fixed axis limits
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)

        
        # Update canvas
        self.fig.tight_layout()

        self.canvas.draw()

def main():
    root = tk.Tk()
    app = DualICRVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main() 