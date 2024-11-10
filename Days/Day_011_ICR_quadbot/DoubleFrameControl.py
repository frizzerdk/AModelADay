import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from ICRVectorControl import calculate_icr_and_omega, calculate_wheel_vectors_from_icr, calculate_wheel_vectors_from_linear_direction



def calculate_center_velocity(global_icr_x, global_icr_y, target_speed=1.0):
    """
    Calculate the velocity of the vehicle's center based on the Global ICR.
    """
    # Handle infinite ICR case (straight line motion)
    if np.isinf(global_icr_x) or np.isinf(global_icr_y):
        # Move in a straight line along the X-axis
        direction = np.array([1.0, 0.0])
    else:
        # Calculate direction perpendicular to the radius from Global ICR to center
        dx = - (0.0 - global_icr_y)
        dy = (0.0 - global_icr_x)
        direction = np.array([dx, dy])
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        else:
            direction = np.array([0.0, 0.0])

    # Velocity vector of the center
    center_velocity = direction * target_speed
    return center_velocity

class DualICRVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual ICR Vector Visualizer")
        
        # Initialize wheel positions relative to vehicle center
        self.wheel_positions = np.array([
            [-0.5, 0.5],   # Front Left
            [0.5, 0.5],    # Front Right
            [-0.5, -0.5],  # Back Left
            [0.5, -0.5]    # Back Right
        ])
        
        # Wheel dimensions
        self.wheel_width = 0.2
        self.wheel_height = 0.1
        
        # Initialize center position
        self.center_position = np.array([0.0, 0.0])
        
        # Initialize show_calc_icr variable
        self.show_calc_icr = tk.BooleanVar(value=True)
        
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
        
        # Global ICR controls
        self.global_icr_x_var = tk.DoubleVar(value=5.0)
        self.global_icr_y_var = tk.DoubleVar(value=np.inf)  # Infinite for straight line
        self.global_speed_var = tk.DoubleVar(value=1.0)
        
        # Local ICR controls
        self.local_icr_x_var = tk.DoubleVar(value=0.0)
        self.local_icr_y_var = tk.DoubleVar(value=0.0)
        self.local_speed_var = tk.DoubleVar(value=1.0)
        
        # Linear term controls
        self.linear_x_var = tk.DoubleVar(value=0.0)
        self.linear_y_var = tk.DoubleVar(value=0.0)
        self.linear_speed_var = tk.DoubleVar(value=0.0)
        
        # Global ICR controls
        ttk.Label(control_frame, text="Global ICR Controls", font='bold').pack()
        ttk.Label(control_frame, text="X:").pack()
        ttk.Scale(control_frame, from_=-10.0, to=10.0, variable=self.global_icr_x_var, 
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Y:").pack()
        ttk.Scale(control_frame, from_=-10.0, to=10.0, variable=self.global_icr_y_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Speed:").pack()
        ttk.Scale(control_frame, from_=0.0, to=3.0, variable=self.global_speed_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Local ICR controls
        ttk.Label(control_frame, text="Local ICR Controls", font='bold').pack()
        ttk.Label(control_frame, text="X:").pack()
        ttk.Scale(control_frame, from_=-2.0, to=2.0, variable=self.local_icr_x_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Y:").pack()
        ttk.Scale(control_frame, from_=-2.0, to=2.0, variable=self.local_icr_y_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Speed:").pack()
        ttk.Scale(control_frame, from_=0.0, to=3.0, variable=self.local_speed_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Linear term controls
        ttk.Label(control_frame, text="Linear Term Controls", font='bold').pack()
        ttk.Label(control_frame, text="X Direction:").pack()
        ttk.Scale(control_frame, from_=-1.0, to=1.0, variable=self.linear_x_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Y Direction:").pack()
        ttk.Scale(control_frame, from_=-1.0, to=1.0, variable=self.linear_y_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        ttk.Label(control_frame, text="Speed:").pack()
        ttk.Scale(control_frame, from_=0.0, to=3.0, variable=self.linear_speed_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Add simulation controls
        sim_frame = ttk.Frame(control_frame)
        sim_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(sim_frame, text="Simulation Controls", font='bold').pack()
        
        # Add simulation time slider
        self.sim_time_var = tk.DoubleVar(value=5.0)
        ttk.Label(sim_frame, text="Simulation Time (s):").pack()
        ttk.Scale(sim_frame, from_=1.0, to=10.0, variable=self.sim_time_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Add simulation step slider
        self.sim_steps_var = tk.IntVar(value=100)
        ttk.Label(sim_frame, text="Simulation Steps:").pack()
        ttk.Scale(sim_frame, from_=50, to=500, variable=self.sim_steps_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Add simulation button
        self.sim_button = ttk.Button(sim_frame, text="Show Trajectory", 
                                   command=self.simulate_trajectory)
        self.sim_button.pack(pady=5)
        
        # Initialize trajectory storage
        self.trajectory_points = None
        
        # Initial plot
        self.update_plot()
        
        # Move the checkbox creation after control_frame is created
        ttk.Checkbutton(control_frame, text="Show Calculated ICR", 
                       variable=self.show_calc_icr,
                       command=self.update_plot).pack()
    
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
        # Get current control inputs
        global_icr_x = self.global_icr_x_var.get()
        global_icr_y = self.global_icr_y_var.get()
        global_speed = self.global_speed_var.get()
        
        local_icr_x = self.local_icr_x_var.get()
        local_icr_y = self.local_icr_y_var.get()
        local_speed = self.local_speed_var.get()
        
        # Get linear term inputs
        linear_x = self.linear_x_var.get()
        linear_y = self.linear_y_var.get()
        linear_speed = self.linear_speed_var.get()
        
        # Normalize linear direction
        linear_dir = np.array([linear_x, linear_y])
        linear_norm = np.linalg.norm(linear_dir)
        if linear_norm > 0:
            linear_dir = linear_dir / linear_norm
        
        # Get simulation parameters
        sim_time = self.sim_time_var.get()
        sim_steps = self.sim_steps_var.get()
        dt = sim_time / sim_steps
        
        # Initialize trajectory storage
        self.trajectory_points = np.zeros((sim_steps, len(self.wheel_positions), 2))
        current_positions = np.copy(self.wheel_positions)
        center_position = np.array([0.0, 0.0])  # Start at origin
        
        # Simulate motion
        for i in range(sim_steps):
            # Calculate center velocity
            center_velocity = calculate_center_velocity(global_icr_x, global_icr_y, global_speed)
            
            # Update center position
            center_position += center_velocity * dt
            
            # Calculate wheel velocities due to Local ICR
            local_wheel_positions = current_positions - center_position - np.array([local_icr_x, local_icr_y])
            local_vectors = calculate_wheel_vectors_from_icr(local_wheel_positions, local_icr_x, local_icr_y, local_speed)
            
            # Add center velocity to each wheel
            global_vectors = np.tile(center_velocity, (len(current_positions), 1))
            
            # Add linear term
            linear_vectors = linear_dir * linear_speed
            combined_vectors = global_vectors + local_vectors + linear_vectors
            
            # Update wheel positions
            current_positions += combined_vectors * dt
            
            # Store current positions
            self.trajectory_points[i] = current_positions
        
        # Update the center position for plotting
        self.center_position = center_position
        
        # Update plot with trajectory
        self.update_plot()
    
    def update_plot(self, *args):
        self.ax.clear()
        # Get current control inputs
        global_icr_x = self.global_icr_x_var.get()
        global_icr_y = self.global_icr_y_var.get()
        global_speed = self.global_speed_var.get()
        
        local_icr_x = self.local_icr_x_var.get()
        local_icr_y = self.local_icr_y_var.get()
        local_speed = self.local_speed_var.get()
        
        # Get linear term inputs
        linear_x = self.linear_x_var.get()
        linear_y = self.linear_y_var.get()
        linear_speed = self.linear_speed_var.get()
        
        # Calculate linear vectors
        linear_dir = np.array([linear_x, linear_y])
        linear_norm = np.linalg.norm(linear_dir)
        if linear_norm > 0:
            linear_dir = linear_dir / linear_norm
        linear_vectors = np.tile(linear_dir * linear_speed, (len(self.wheel_positions), 1))
        
        # Calculate center velocity
        center_velocity = calculate_center_velocity(global_icr_x, global_icr_y, global_speed)
        
        # Calculate wheel velocities due to Local ICR
        local_wheel_positions = self.wheel_positions - np.array([local_icr_x, local_icr_y])
        local_vectors = calculate_wheel_vectors_from_icr(local_wheel_positions, local_icr_x, local_icr_y, local_speed)
        
        # Add center velocity to each wheel
        global_vectors = np.tile(center_velocity, (len(self.wheel_positions), 1))
        
        # Combine the velocities
        combined_vectors = global_vectors + local_vectors + linear_vectors
        
        # Calculate angles for wheel display from combined vectors
        angles = np.arctan2(combined_vectors[:, 1], combined_vectors[:, 0])
        speeds = np.linalg.norm(combined_vectors, axis=1)
        
        # Shift wheel positions by the center position
        shifted_wheel_positions = self.wheel_positions + self.center_position
        
        # Plot robot frame
        self.ax.plot(shifted_wheel_positions[[0,1,3,2,0], 0], 
                    shifted_wheel_positions[[0,1,3,2,0], 1], 
                    'k-', label='Robot Frame')
        
        # Plot wheels and their vectors
        colors = ['blue', 'red', 'green', 'purple']
        labels = ['FL', 'FR', 'BL', 'BR']
        
        for i, (pos, vec, angle, speed, color, label) in enumerate(
                zip(shifted_wheel_positions, combined_vectors, 
                    angles, speeds, colors, labels)):
            
            # Draw wheel rectangle
            wheel = self.draw_wheel(pos, angle, color)
            self.ax.add_patch(wheel)
            
            # Plot combined vector (thicker)
            self.ax.arrow(pos[0], pos[1], 
                         vec[0]*0.2, vec[1]*0.2,
                         head_width=0.05, head_length=0.1,
                         fc=color, ec=color,
                         label=f'{label} wheel (speed={speed:.2f})')
        
        # Plot Global ICR point
        self.ax.plot(global_icr_x, global_icr_y, 'rx', markersize=10, 
                     label=f'Global ICR ({global_icr_x:.1f}, {global_icr_y:.1f})')
        
        # Plot Local ICR point (relative to vehicle center)
        local_icr_global_x = self.center_position[0] + local_icr_x
        local_icr_global_y = self.center_position[1] + local_icr_y
        self.ax.plot(local_icr_global_x, local_icr_global_y, 'bx', markersize=10, 
                     label=f'Local ICR ({local_icr_global_x:.1f}, {local_icr_global_y:.1f})')
        
        # Plot vehicle center
        self.ax.plot(self.center_position[0], self.center_position[1], 'ko', markersize=8, label='Vehicle Center')
        
        # Plot trajectories if they exist
        if self.trajectory_points is not None:
            for wheel_idx in range(len(self.wheel_positions)):
                trajectory = self.trajectory_points[:, wheel_idx]
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], '--', 
                           color=colors[wheel_idx], alpha=0.5,
                           label=f'{labels[wheel_idx]} trajectory')
            # Plot center trajectory
            center_trajectory = self.trajectory_points.mean(axis=1)
            self.ax.plot(center_trajectory[:, 0], center_trajectory[:, 1], 'k--', alpha=0.7, label='Center Trajectory')
        
        # Set plot properties
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_title('Combined ICR Vector Fields')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        
        # Set fixed axis limits
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        
        # After calculating combined vectors but before plotting:
        if self.show_calc_icr.get():
            # Calculate ICR for each pair of wheels
            pairs = [(0,1), (1,3), (3,2), (2,0)]  # FL-FR, FR-BR, BR-BL, BL-FL
            for i, (idx1, idx2) in enumerate(pairs):
                pos1 = shifted_wheel_positions[idx1]
                pos2 = shifted_wheel_positions[idx2]
                vel1 = combined_vectors[idx1]
                vel2 = combined_vectors[idx2]
                
                icr, omega1, omega2 = calculate_icr_and_omega(pos1, vel1, pos2, vel2)
                
                if icr is not None:
                    # Plot calculated ICR point
                    self.ax.plot(icr[0], icr[1], 'g+', markersize=8, 
                               label=f'Calc ICR {i+1}' if i == 0 else '')
                    
                    # Add omega annotations near the wheels
                    if omega1 is not None:
                        self.ax.annotate(f'ω={omega1:.2f}', 
                                       xy=(pos1[0], pos1[1]),
                                       xytext=(10, 10), 
                                       textcoords='offset points')
                    if omega2 is not None:
                        self.ax.annotate(f'ω={omega2:.2f}', 
                                       xy=(pos2[0], pos2[1]),
                                       xytext=(10, -10), 
                                       textcoords='offset points')
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = DualICRVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
