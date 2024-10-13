import xml.etree.ElementTree as ET
import os
import mujoco
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
