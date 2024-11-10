import numpy as np

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

def rotate_points(points, angle):
    """Rotate points around origin by angle (in radians)"""
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(points, rotation_matrix.T)

def calculate_wheel_vectors_from_icr(wheel_positions,
                                      icr_x, icr_y, 
                                      target_speed=1.0, 
                                      wheel_rotation=0.0, 
                                      added_noise=0.001):
    """
    Calculate steering angles and wheel speeds for each wheel based on ICR position.
    
    Args:
        wheel_positions: List of (x,y) coordinates for each wheel
        icr_x: X coordinate of Instantaneous Center of Rotation
        icr_y: Y coordinate of Instantaneous Center of Rotation
        target_speed: Target average wheel speed (default 1.0)
        wheel_rotation: Rotation of wheel positions in local frame (radians)
    
    Returns:
        angles: Array of steering angles for each wheel (in radians)
        speeds: Array of wheel speeds
        vectors: Array of (x,y) vectors for each wheel
        radii: Array of radii from ICR to each wheel
        rotated_positions: Array of rotated wheel positions
    """
    wheel_positions = np.array(wheel_positions)
    
    # Rotate wheel positions in local frame
    rotated_positions = rotate_points(wheel_positions, wheel_rotation)
    
    # Handle infinite ICR case (straight line motion)
    if np.isinf(icr_x) or np.isinf(icr_y):
        angles = np.zeros(len(rotated_positions))
        speeds = np.ones(len(rotated_positions)) * target_speed
        vectors = np.array([
            np.cos(angles),
            np.sin(angles)
        ]).T * speeds[:, np.newaxis]
        radii = np.full(len(rotated_positions), np.inf)
        return angles, speeds, vectors, radii, rotated_positions
    
    # Calculate angles to ICR (for wheel orientation, add π/2 for tangent)
    angles = np.arctan2(
        rotated_positions[:, 1] - icr_y,
        rotated_positions[:, 0] - icr_x
    )+ np.pi/2  # Add 90 degrees to make wheels tangent to circles
    
    # Calculate radii from ICR to each wheel
    radii = np.sqrt(
        (rotated_positions[:, 0] - icr_x)**2 + 
        (rotated_positions[:, 1] - icr_y)**2
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
    if added_noise > 0:
        vectors += np.random.randn(*vectors.shape)*added_noise
    return vectors
    
def calculate_wheel_vectors_from_linear_direction(wheel_positions, speed_vector, wheel_rotation=0.0):
    """Calculate wheel vectors for linear motion"""
    # copy speed vector for each wheel with same direction and magnitude
    vectors = np.tile(speed_vector, (len(wheel_positions), 1))
    # rotate wheel positions
    vectors = rotate_points(wheel_positions, wheel_rotation)

    return vectors