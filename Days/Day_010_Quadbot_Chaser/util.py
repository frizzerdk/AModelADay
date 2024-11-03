import xml.etree.ElementTree as ET
import os
import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

def modify_xml_parameter(xml_string, element_name, attribute, value):
    # Parse the XML
    root = ET.fromstring(xml_string)

    # Find the element
    element = root.find(f".//*[@name='{element_name}']")
    
    if element is not None:
        # Modify the attribute
        element.set(attribute, str(value))
        print(f"Modified {element_name}: set {attribute} to {value}")
    else:
        print(f"Element with name '{element_name}' not found")

    # Convert modified XML back to string
    return ET.tostring(root, encoding="unicode")

def modify_xml_parameters(xml_string, parameters):
    """
    Modify multiple parameters in the XML string.
    Parameters:
        xml_string (str): The XML string to modify.
        parameters (list): A list of tuples, where each tuple contains (element_name, attribute, value).
    Returns:
        str: The modified XML string.
    """
    for element_name, attribute, value in parameters:
        xml_string = modify_xml_parameter(xml_string, element_name, attribute, value)
    return xml_string

def make_modified_xml_file(xml_file_in, xml_file_out, parameters):
    try:
        with open(xml_file_in, "r") as file:
            xml_string = file.read()
    except FileNotFoundError:
        current_dir = os.getcwd()
        print(f"Error: File '{xml_file_in}' not found.")
        print(f"Current working directory: {current_dir}")
        print(f"Absolute path attempted: {os.path.abspath(xml_file_in)}")
        print("Please ensure the XML file is in the correct location.")
        raise

    modified_xml_string = modify_xml_parameters(xml_string, parameters)
    
    try:
        with open(xml_file_out, "w") as file:
            file.write(modified_xml_string)
    except IOError as e:
        print(f"Error writing to file '{xml_file_out}': {e}")
        raise

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


def plot_summary(episode_data=None, file_path=None, target_pos=None):
    '''
    Plot the trajectories of the quadbot
    episode_data: list of dicts, each dict is a log_data from the env for a single episode
    file_path: path to save the plot to, if None, a new file is created
    '''
    if not isinstance(episode_data, list):
        episode_data = [episode_data]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calculate opacity steps
    n_episodes = len(episode_data)
    opacity_range = np.linspace(0.5, 1.0, n_episodes)
    
    # Create a custom colormap
    colors = ['blue', 'green', 'yellow', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    # Plot each episode
    for episode, opacity in zip(episode_data, opacity_range):
        x_positions = episode['sensor/body_pos_x']
        y_positions = episode['sensor/body_pos_y']
        x_velocities = episode['sensor/body_linvel_x']
        y_velocities = episode['sensor/body_linvel_y']
        speeds = np.sqrt(np.array(x_velocities)**2 + np.array(y_velocities)**2)
        
        # Plot the trajectory with increasing opacity
        scatter = ax.scatter(x_positions, y_positions, c=speeds, 
                           cmap=cmap, s=10, alpha=opacity)
        
        # Plot start and end points for each trajectory
        ax.plot(x_positions[0], y_positions[0], 'go', markersize=10, 
                alpha=opacity, label=f'Start {n_episodes-1}' if opacity == 1.0 else None)
        ax.plot(x_positions[-1], y_positions[-1], 'ro', markersize=10, 
                alpha=opacity, label=f'End {n_episodes-1}' if opacity == 1.0 else None)

    # Calculate plot limits based on start points and target
    all_x = [episode['sensor/body_pos_x'][0] for episode in episode_data]  # Start x positions
    all_y = [episode['sensor/body_pos_y'][0] for episode in episode_data]  # Start y positions
    
    # Add target position to limits calculation if available
    if target_pos is not None:
        all_x.append(target_pos[0])
        all_y.append(target_pos[1])
    
    # Calculate min and max with 50% margin
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    x_margin = (x_max - x_min) * 0.5
    y_margin = (y_max - y_min) * 0.5
    
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    # Plot target position (assuming it's the same for all episodes)
    if target_pos is not None:
        ax.plot(target_pos[0], target_pos[1], 
                'y*', markersize=20, label='Target')

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Speed')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Quadbot Trajectories')
    ax.legend()

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Ensure equal aspect ratio
    ax.set_aspect('equal', 'box')
    
    # Save the plot
    if not file_path:
        if not os.path.exists("trajectory"):
            os.makedirs("trajectory")
        file_path = f"trajectory/quadbot_trajectory_{time.time()}.jpeg"
    
    plt.savefig(file_path, format='jpeg', dpi=150, 
                bbox_inches='tight', pil_kwargs={'quality': 10})
    plt.close(fig)

    return file_path
