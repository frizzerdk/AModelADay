import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as transforms
from ICRVectorControl import calculate_icr_and_omega

def calculate_wheel_vectors(wheel_positions, v_linear, omega, angle, target_speed=1.0):
    """
    Calculate the velocity vector for each wheel based on linear and angular velocities.
    
    Args:
        wheel_positions: List of (x,y) coordinates for each wheel
        v_linear: Linear velocity magnitude
        omega: Angular velocity (rad/s)
        angle: Direction of linear velocity (rad)
        target_speed: Target average wheel speed (default 1.0)
    """
    wheel_positions = np.array(wheel_positions)
    
    # Initialize arrays
    vectors = np.zeros_like(wheel_positions)
    
    # Calculate linear velocity components
    v_linear_x = v_linear * np.cos(angle)
    v_linear_y = v_linear * np.sin(angle)
    
    for i, pos in enumerate(wheel_positions):
        # Linear component (now with direction)
        v_linear_component = np.array([v_linear_x, v_linear_y])
        
        # Rotational component (perpendicular to radius vector)
        r = pos  # Radius vector from center
        v_rotational_component = np.array([-omega * r[1], omega * r[0]])
        
        # Total wheel velocity
        vectors[i] = v_linear_component + v_rotational_component
    
    # Calculate angles and speeds
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    speeds = np.linalg.norm(vectors, axis=1)
    
    # Normalize speeds to maintain target average speed
    if np.mean(speeds) > 0:
        vectors = vectors / np.mean(speeds) * target_speed
        speeds = np.linalg.norm(vectors, axis=1)
    
    return angles, speeds, vectors

class LinearOmegaVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear-Omega Vector Visualizer")
        
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
        
        # Create sliders
        self.v_linear_var = tk.DoubleVar(value=1.0)
        self.omega_var = tk.DoubleVar(value=0.0)
        self.speed_var = tk.DoubleVar(value=1.0)
        
        # Add direction slider
        self.direction_var = tk.DoubleVar(value=0.0)
        
        # Linear velocity slider
        ttk.Label(control_frame, text="Linear Velocity:").pack()
        ttk.Scale(control_frame, from_=-3.0, to=3.0, variable=self.v_linear_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Direction slider (in degrees for user-friendliness)
        ttk.Label(control_frame, text="Direction (degrees):").pack()
        ttk.Scale(control_frame, from_=-180, to=180, variable=self.direction_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Angular velocity slider
        ttk.Label(control_frame, text="Angular Velocity (rad/s):").pack()
        ttk.Scale(control_frame, from_=-3.0, to=3.0, variable=self.omega_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
        # Speed scale slider
        ttk.Label(control_frame, text="Speed Scale:").pack()
        ttk.Scale(control_frame, from_=0.1, to=3.0, variable=self.speed_var,
                 orient=tk.HORIZONTAL, command=self.update_plot).pack(fill=tk.X)
        
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
    
    def calculate_icr(self, vectors):
        """Calculate ICR from wheel vectors"""
        # Use first two wheels to calculate ICR
        pos1, pos2 = self.wheel_positions[0], self.wheel_positions[1]
        vel1, vel2 = vectors[0], vectors[1]
        
        icr, omega1, omega2 = calculate_icr_and_omega(pos1, vel1, pos2, vel2)
        return icr, omega1, omega2
    
    def update_plot(self, *args):
        self.ax.clear()
        
        # Get current values
        v_linear = self.v_linear_var.get()
        omega = self.omega_var.get()
        target_speed = self.speed_var.get()
        direction = np.radians(self.direction_var.get())  # Convert degrees to radians
        
        # Calculate vectors with direction
        angles, speeds, vectors = calculate_wheel_vectors(
            self.wheel_positions, v_linear, omega, direction, target_speed
        )
        
        # Plot direction arrow from center
        if v_linear != 0:
            self.ax.arrow(0, 0, 
                         0.5 * np.cos(direction), 0.5 * np.sin(direction),
                         head_width=0.05, head_length=0.1,
                         fc='yellow', ec='yellow',
                         label=f'Direction ({self.direction_var.get():.1f}°)')
        
        # Calculate ICR
        icr, omega1, omega2 = self.calculate_icr(vectors)
        
        # Plot robot frame
        self.ax.plot(self.wheel_positions[[0,1,3,2,0], 0],
                    self.wheel_positions[[0,1,3,2,0], 1],
                    'k-', label='Robot Frame')
        
        # Plot wheels and their vectors
        colors = ['blue', 'red', 'green', 'purple']
        labels = ['FL', 'FR', 'BL', 'BR']
        
        for i, (pos, vec, angle, speed, color, label) in enumerate(
                zip(self.wheel_positions, vectors, angles, speeds, colors, labels)):
            
            # Draw wheel rectangle
            wheel = self.draw_wheel(pos, angle, color)
            self.ax.add_patch(wheel)
            
            # Plot velocity vector
            self.ax.arrow(pos[0], pos[1],
                         vec[0]*0.2, vec[1]*0.2,
                         head_width=0.05, head_length=0.1,
                         fc=color, ec=color,
                         label=f'{label} wheel (speed={speed:.2f})')
        
        # Plot ICR if it exists
        if icr is not None and not (np.isinf(icr[0]) or np.isinf(icr[1])):
            self.ax.plot(icr[0], icr[1], 'ko', markersize=10, label='ICR')
            # Draw lines from ICR to wheels
            for pos, color in zip(self.wheel_positions, colors):
                self.ax.plot([icr[0], pos[0]], [icr[1], pos[1]], 
                           color=color, linestyle=':', alpha=0.5)
            
            # Add ICR coordinates to title
            title = (f'Linear-Omega Control (v={v_linear:.2f}, dir={self.direction_var.get():.1f}°, ω={omega:.2f})\n'
                    f'ICR: ({icr[0]:.2f}, {icr[1]:.2f})')
        else:
            title = (f'Linear-Omega Control (v={v_linear:.2f}, dir={self.direction_var.get():.1f}°, ω={omega:.2f})\n'
                    f'ICR: Infinite (Linear Motion)')
        
        # Set plot properties
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_title(title)
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
    app = LinearOmegaVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
