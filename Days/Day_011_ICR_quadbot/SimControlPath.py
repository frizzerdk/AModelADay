from abc import ABC, abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
from ICRVectorControl import calculate_wheel_vectors, rotate_points, calculate_icr_and_omega



ControlPath = []

class ControlInput(ABC):
    @abstractmethod
    def get_control_input(self,t:float)->dict:
        """
        Get the control input at time t and return a dictionary of wheel vectors in the global Local Frame
        """
        pass


class SimpleICRControlInput(ControlInput):
    def __init__(self,velocity,ICR_x,ICR_y):
        self.velocity = velocity
        self.ICR_x = ICR_x
        self.ICR_y = ICR_y
        self.wheel_positions = np.array([
            [-0.5, 0.5],   # Front Left
            [0.5, 0.5],    # Front Right
            [-0.5, -0.5],  # Back Left
            [0.5, -0.5]    # Back Right
        ])
    
    def get_control_input(self,t:float)->dict:
        """
        Get the control input at time t and return a dictionary of wheel vectors
        """
        vectors = calculate_wheel_vectors(
            self.wheel_positions, self.ICR_x, self.ICR_y, self.velocity, 0
        )
        return {
            'vectors': vectors
        }

@jax.jit
def global2local_from_global_position(global_position):
    """
    Get the local frame from the global position
    """
    # get frame angle
    front_vector = global_position[1] - global_position[0]  # FR - FL
    frame_angle = jnp.arctan2(front_vector[1], front_vector[0])
    
    # Create rotation matrix
    cos_theta = jnp.cos(frame_angle)
    sin_theta = jnp.sin(frame_angle)
    rotation_matrix = jnp.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    return rotation_matrix

@jax.jit
def local2global(position,vector):
    rotation_matrix = global2local_from_global_position(position)
    return jnp.einsum('ij,kj->ki',rotation_matrix, vector)

def SimulateControlPath(ControlInput:ControlInput, dt:float, t_max:float)->list[dict]:
    """
    Simulate the control path using JAX-based RK4 with geometric constraints
    """
    @jax.jit
    def rk4_step(state, t, dt):
        """Perform single RK4 integration step"""
        k1 = system_delta(state, t)
        k2 = system_delta(state + dt * k1 / 2, t + dt / 2)
        k3 = system_delta(state + dt * k2 / 2, t + dt / 2)
        k4 = system_delta(state + dt * k3, t + dt)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @jax.jit
    def system_delta(state, t):
        """System dynamics in state space"""
        positions = state.reshape((4, 2))  # 4 wheels x 2 coordinates

        # Get control input velocities
        velocities = jnp.array(ControlInput.get_control_input(t)['vectors'])
        
        # Apply rotation to each velocity vector
        velocities_rotated = local2global(positions,velocities)
        
        # Return state derivatives [velocities]
        return velocities_rotated.flatten()

    @jax.jit
    def geometric_constraints(state):
        """Compute geometric constraint violations"""
        positions = state.reshape((4, 2))
        
        # Compute wheel distances
        fl, fr, bl, br = positions[0], positions[1], positions[2], positions[3]
        
        # Desired distances (based on initial configuration)
        target_width = 1.0   # Distance between left and right wheels
        target_length = 1.0  # Distance between front and back wheels
        target_diag = jnp.sqrt(target_width**2 + target_length**2)
        
        # Actual distances
        width_front = jnp.linalg.norm(fl - fr)
        width_back = jnp.linalg.norm(bl - br)
        length_left = jnp.linalg.norm(fl - bl)
        length_right = jnp.linalg.norm(fr - br)
        diag1 = jnp.linalg.norm(fl - br)
        diag2 = jnp.linalg.norm(fr - bl)
        
        # Constraint violations
        violations = jnp.array([
            width_front - target_width,
            width_back - target_width,
            length_left - target_length,
            length_right - target_length,
            diag1 - target_diag,
            diag2 - target_diag
        ])
        
        return jnp.sum(violations**2)  # Return total squared violation

    @partial(jax.jit, static_argnums=(1,))
    def project_to_constraints(state, num_iterations=5):
        """Project state back onto constraint manifold using gradient descent"""
        def loss(s):
            return geometric_constraints(s)
        
        grad_fn = jax.grad(loss)
        current_state = state
        
        # Simple gradient descent
        for _ in range(num_iterations):
            gradient = grad_fn(current_state)
            current_state = current_state - 0.01 * gradient
        
        return current_state

    # Initialize state with actual wheel positions instead of zeros
    initial_state = jnp.array([
        [-0.5, 0.5],   # Front Left
        [0.5, 0.5],    # Front Right
        [-0.5, -0.5],  # Back Left
        [0.5, -0.5]    # Back Right
    ]).flatten()  # Convert to 1D array of shape (8,)

    # Simulation loop
    times = jnp.arange(0, t_max, dt)
    states = []
    violations = []
    current_state = initial_state
    
    for t in times:
        # RK4 step
        current_state = rk4_step(current_state, t, dt)
        
        # Project back to constraints
        current_state = project_to_constraints(current_state)
        
        # Track state and violations
        states.append(current_state)
        violations.append(geometric_constraints(current_state))
    
    # Convert to control path format
    control_path = []
    for t, state in zip(times, states):
        control = ControlInput.get_control_input(float(t))
        control_path.append({
            'state': jax.device_get(state),  # Convert back to numpy
            'constraint_violation': float(jax.device_get(violations[-1])),
            **control
        })
    
    return control_path

def PlotControlPath(control_path: list[dict]):
    """
    Plot the control path showing robot position and wheel configurations over time
    """

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot robot path and wheel positions
    wheel_positions = np.array([
        [-0.5, 0.5],   # Front Left
        [0.5, 0.5],    # Front Right
        [-0.5, -0.5],  # Back Left
        [0.5, -0.5]    # Back Right
    ])
    
    # Plot constraint violations
    violations = [step['constraint_violation'] for step in control_path]
    times = np.arange(len(violations)) * 0.01  # Assuming dt=0.01
    ax2.plot(times, violations, label='Constraint Violations')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Constraint Violation')
    ax2.set_title('Geometric Constraint Violations')
    ax2.grid(True)
    
    # Plot robot configurations at regular intervals
    plot_interval = max(1, len(control_path) // 20)  # Plot ~20 configurations
    wheel_width = 0.2
    wheel_height = 0.1
    
    for i, step in enumerate(control_path[::plot_interval]):
        state = step['state']
        positions = state.reshape((4, 2))
        
        # Plot robot frame
        frame = positions[[0,1,3,2,0], :]
        ax1.plot(frame[:, 0], frame[:, 1], 'k-', alpha=0.3)
        
        # Plot wheels with corrected vector indexing
        for wheel_idx, pos in enumerate(positions):
            # Create wheel rectangle
            rect = Rectangle(
                (pos[0] - wheel_width/2, pos[1] - wheel_height/2),
                wheel_width, wheel_height,
                facecolor='blue', alpha=0.3
            )
            

            # Plot velocity vector using wheel_idx directly
            vector = step['vectors'][wheel_idx]
            # Add vector as a new axis to match expected dimensions
            vector_expanded = jnp.expand_dims(vector, 0)
            rotated_vector = local2global(positions, vector_expanded)[0]  # Take first (only) result
            ax1.arrow(pos[0], pos[1], 
                     rotated_vector[0]*0.2, rotated_vector[1]*0.2,
                     head_width=0.05, 
                     head_length=0.1,
                     fc='red', 
                     ec='red',
                     alpha=0.3)
            
            # calculate angle of wheel
            angle = np.arctan2(vector[1],vector[0])
            rect.set_transform(transforms.Affine2D().rotate_around(pos[0], pos[1], angle))
            ax1.add_patch(rect)

    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Robot Path and Configurations')
    ax1.axis('equal')
    ax1.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_init_control_input(control_input:ControlInput):
    """
    Plot the initial control input
    """
    # Create a range of time points
    t_points = np.linspace(0, 10.0, 20)  # 20 points from 0 to 10 seconds
    
    plt.figure(figsize=(10,10))
    
    wheel_positions = np.array([
        [-0.5, 0.5],   # Front Left
        [0.5, 0.5],    # Front Right
        [-0.5, -0.5],  # Back Left
        [0.5, -0.5]    # Back Right
    ])

    # Plot robot frame
    frame = wheel_positions[[0,1,3,2,0], :]
    plt.plot(frame[:, 0], frame[:, 1], 'k-', alpha=0.3)

    # Plot control vectors at each time point
    for t in t_points:
        control = control_input.get_control_input(t)
        vectors = control['vectors']
        
        for wheel_idx, pos in enumerate(wheel_positions):
            vector = vectors[wheel_idx]
            plt.arrow(pos[0], pos[1], 
                     vector[0]*0.2, vector[1]*0.2,
                     head_width=0.05, 
                     head_length=0.1,
                     fc='red', 
                     ec='red',
                     alpha=0.1)  # Reduced alpha for clearer visualization
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('Control Vectors Over Time')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show(block=False)

def calculate_icr_and_omega(pos1, vel1, pos2, vel2):
    """
    Calculate the ICR (Instantaneous Center of Rotation) and angular velocities (omega)
    for two wheels based on their position and velocity vectors.
    
    Parameters:
    pos1, pos2: array-like, shape (2,)
        Position vectors (x, y) of wheel 1 and wheel 2 respectively.
    vel1, vel2: array-like, shape (2,)
        Velocity vectors (vx, vy) of wheel 1 and wheel 2 respectively.
        
    Returns:
    ICR: tuple (float, float)
        Coordinates (x, y) of the Instantaneous Center of Rotation.
    omega1, omega2: float
        Angular velocities for wheel 1 and wheel 2 around the ICR.
    """
    # Convert inputs to numpy arrays
    pos1, vel1 = np.array(pos1), np.array(vel1)
    pos2, vel2 = np.array(pos2), np.array(vel2)

    # Calculate the slopes (dx/dy) for perpendicular lines
    # The perpendicular direction of a vector (vx, vy) is (-vy, vx)
    perp_vel1 = np.array([-vel1[1], vel1[0]])
    perp_vel2 = np.array([-vel2[1], vel2[0]])

    # Set up the linear equations for the intersection point (ICR)
    # pos1 + t1 * perp_vel1 = pos2 + t2 * perp_vel2
    # Rearranged: t1 * perp_vel1 - t2 * perp_vel2 = pos2 - pos1
    A = np.array([perp_vel1, -perp_vel2]).T
    b = pos2 - pos1

    # Solve for t1 and t2
    try:
        t1, t2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("No unique ICR found. Wheels are either parallel or collinear.")
        return None, None, None

    # Calculate the ICR position
    ICR = pos1 + t1 * perp_vel1

    # Calculate the distance of each wheel to the ICR
    r1 = np.linalg.norm(ICR - pos1)
    r2 = np.linalg.norm(ICR - pos2)

    # Calculate the angular velocities (omega = speed / radius)
    speed1 = np.linalg.norm(vel1)
    speed2 = np.linalg.norm(vel2)
    
    omega1 = speed1 / r1 if r1 != 0 else 0
    omega2 = speed2 / r2 if r2 != 0 else 0

    return ICR, omega1, omega2

def plot_icr_calculation(control_input: ControlInput):
    """
    Plot the ICR calculation for the initial control input
    """
    plt.figure(figsize=(10, 10))
    
    # Get initial control vectors
    control = control_input.get_control_input(0)
    vectors = control['vectors']
    
    wheel_positions = np.array([
        [-0.5, 0.5],   # Front Left
        [0.5, 0.5],    # Front Right
        [-0.5, -0.5],  # Back Left
        [0.5, -0.5]    # Back Right
    ])

    # Plot robot frame
    frame = wheel_positions[[0,1,3,2,0], :]
    plt.plot(frame[:, 0], frame[:, 1], 'k-', label='Robot Frame')
    
    # Calculate and plot ICR for each pair of wheels
    wheel_pairs = [(0,1), (2,3), (0,2), (1,3)]  # FL-FR, BL-BR, FL-BL, FR-BR
    colors = ['red', 'blue', 'green', 'purple']
    
    for (w1, w2), color in zip(wheel_pairs, colors):
        icr, omega1, omega2 = calculate_icr_and_omega(
            wheel_positions[w1], vectors[w1],
            wheel_positions[w2], vectors[w2]
        )
        
        if icr is not None:
            # Plot ICR point
            plt.plot(icr[0], icr[1], 'o', color=color, 
                    label=f'ICR {w1}-{w2}')
            
            # Plot lines from wheels to ICR
            plt.plot([wheel_positions[w1,0], icr[0]], 
                    [wheel_positions[w1,1], icr[1]], 
                    '--', color=color, alpha=0.3)
            plt.plot([wheel_positions[w2,0], icr[0]], 
                    [wheel_positions[w2,1], icr[1]], 
                    '--', color=color, alpha=0.3)
    
    # Plot wheel velocity vectors
    for i, (pos, vec) in enumerate(zip(wheel_positions, vectors)):
        plt.arrow(pos[0], pos[1], 
                 vec[0]*0.2, vec[1]*0.2,
                 head_width=0.05, 
                 head_length=0.1,
                 fc='black', 
                 ec='black',
                 label=f'Wheel {i} velocity')
    
    plt.axis('equal')
    plt.grid(True)
    plt.title('ICR Calculations for Wheel Pairs')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.show(block=False)

if __name__ == "__main__":
    control_input = SimpleICRControlInput(velocity=1.0, ICR_x=1.0, ICR_y=1.0)
    plot_init_control_input(control_input)
    plot_icr_calculation(control_input)
    control_path = SimulateControlPath(control_input, dt=0.01, t_max=10.0)
    PlotControlPath(control_path)
    