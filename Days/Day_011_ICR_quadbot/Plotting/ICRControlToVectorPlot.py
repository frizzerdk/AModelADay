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
    """
    Calculate steering angles and wheel speeds for each wheel based on ICR position.
    
    Args:
        wheel_positions: List of (x,y) coordinates for each wheel
        icr_x: X coordinate of Instantaneous Center of Rotation
        icr_y: Y coordinate of Instantaneous Center of Rotation
        target_speed: Target average wheel speed (default 1.0)
    
    Returns:
        angles: Array of steering angles for each wheel (in radians)
        speeds: Array of wheel speeds
        vectors: Array of (x,y) vectors for each wheel
        radii: Array of radii from ICR to each wheel
    """
    wheel_positions = np.array(wheel_positions)
    
    # Handle infinite ICR case (straight line motion)
    if np.isinf(icr_x) or np.isinf(icr_y):
        angles = np.zeros(len(wheel_positions))
        speeds = np.ones(len(wheel_positions)) * target_speed
        vectors = np.array([
            np.cos(angles),
            np.sin(angles)
        ]).T * speeds[:, np.newaxis]
        radii = np.full(len(wheel_positions), np.inf)
        return angles, speeds, vectors, radii
    
    # Calculate angles to ICR (for wheel orientation, add π/2 for tangent)
    angles = np.arctan2(
        wheel_positions[:, 1] - icr_y,
        wheel_positions[:, 0] - icr_x
    ) + np.pi/2  # Add 90 degrees to make wheels tangent to circles
    
    # Calculate radii from ICR to each wheel
    radii = np.sqrt(
        (wheel_positions[:, 0] - icr_x)**2 + 
        (wheel_positions[:, 1] - icr_y)**2
    )
    
    # Normalize speeds to maintain target average speed
    mean_radius = np.mean(radii)
    if mean_radius > 0:
        speeds = radii / mean_radius * target_speed
    else:
        speeds = np.ones_like(radii) * target_speed
    
    # Calculate vectors (tangent to circles)
    vectors = np.array([
        np.cos(angles),
        np.sin(angles)
    ]).T * speeds[:, np.newaxis]
    
    return angles, speeds, vectors, radii

class ICRVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("ICR Wheel Vector Visualizer")
        
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
        self.icr_x_var = tk.DoubleVar(value=1.0)
        self.icr_y_var = tk.DoubleVar(value=1.0)
        self.speed_var = tk.DoubleVar(value=1.0)
        
        # ICR X slider
        ttk.Label(control_frame, text="ICR X:").pack()
        self.icr_x_slider = ttk.Scale(control_frame, from_=-5.0, to=5.0, 
                                    variable=self.icr_x_var, orient=tk.HORIZONTAL,
                                    command=self.update_plot)
        self.icr_x_slider.pack(fill=tk.X)
        
        # ICR Y slider
        ttk.Label(control_frame, text="ICR Y:").pack()
        self.icr_y_slider = ttk.Scale(control_frame, from_=-5.0, to=5.0,
                                    variable=self.icr_y_var, orient=tk.HORIZONTAL,
                                    command=self.update_plot)
        self.icr_y_slider.pack(fill=tk.X)
        
        # Speed slider
        ttk.Label(control_frame, text="Target Speed:").pack()
        self.speed_slider = ttk.Scale(control_frame, from_=0.1, to=3.0,
                                    variable=self.speed_var, orient=tk.HORIZONTAL,
                                    command=self.update_plot)
        self.speed_slider.pack(fill=tk.X)
        
        # Initial plot
        self.update_plot()
    
    def draw_wheel(self, pos, angle, color):
        """Draw a wheel as a rotated rectangle"""
        # Create a rectangle
        rect = Rectangle(
            (pos[0] - self.wheel_width/2, pos[1] - self.wheel_height/2),
            self.wheel_width, self.wheel_height,
            facecolor=color, alpha=0.5, edgecolor='black'
        )
        
        # Create transform for rotation around wheel center
        t = transforms.Affine2D().rotate_around(pos[0], pos[1], angle)
        rect.set_transform(t + self.ax.transData)
        
        return rect
    
    def draw_wheel_path(self, pos, icr_x, icr_y, color):
        """Draw the circular path that the wheel would follow"""
        if np.isinf(icr_x) or np.isinf(icr_y):
            # Draw straight line for infinite ICR
            x_min, x_max = self.ax.get_xlim()
            y = pos[1]
            self.ax.plot([x_min, x_max], [y, y], '--', color=color, alpha=0.3)
        else:
            # Calculate radius
            radius = np.sqrt((pos[0] - icr_x)**2 + (pos[1] - icr_y)**2)
            # Create circle patch
            circle = Circle((icr_x, icr_y), radius, fill=False, linestyle='--', 
                          color=color, alpha=0.3)
            self.ax.add_patch(circle)
        
    def update_plot(self, *args):
        self.ax.clear()
        
        # Get current values
        icr_x = self.icr_x_var.get()
        icr_y = self.icr_y_var.get()
        target_speed = self.speed_var.get()
        
        # Calculate vectors
        angles, speeds, vectors, radii = calculate_wheel_vectors(
            self.wheel_positions, icr_x, icr_y, target_speed
        )
        
        # Plot robot frame
        self.ax.plot(self.wheel_positions[[0,1,3,2,0], 0], 
                    self.wheel_positions[[0,1,3,2,0], 1], 
                    'k-', label='Robot Frame')
        
        # Plot wheels and their vectors
        colors = ['blue', 'red', 'green', 'purple']
        labels = ['FL', 'FR', 'BL', 'BR']
        
        for i, (pos, vec, angle, speed, radius, color, label) in enumerate(
                zip(self.wheel_positions, vectors, angles, speeds, radii, colors, labels)):
            # Draw wheel path
            self.draw_wheel_path(pos, icr_x, icr_y, color)
            
            # Draw wheel rectangle
            wheel = self.draw_wheel(pos, angle, color)
            self.ax.add_patch(wheel)
            
            # Plot velocity vector
            self.ax.arrow(pos[0], pos[1], 
                         vec[0]*0.2, vec[1]*0.2,
                         head_width=0.05, 
                         head_length=0.1, 
                         fc=color, 
                         ec=color,
                         label=f'{label} wheel (speed={speed:.2f}, r={radius:.2f})')
        
        # Plot ICR point
        if not (np.isinf(icr_x) or np.isinf(icr_y)):
            self.ax.plot(icr_x, icr_y, 'rx', markersize=10, 
                        label=f'ICR ({icr_x:.1f}, {icr_y:.1f})')
        
        # Set plot properties
        self.ax.axis('equal')
        self.ax.grid(True)
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.set_title('Robot Configuration with Wheel Vectors')
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
    app = ICRVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
