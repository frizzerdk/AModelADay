import os
# change dir to dir of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import mujoco
import mujoco.viewer
import time
import numpy as np
model_path = 'cartpole.xml'
# load XML from file
with open(model_path, 'r') as f:
  XML = f.read()

ASSETS=dict()
with open('body.stl', 'rb') as f:
  ASSETS['body.stl'] = f.read()

def print_model_info(model_path):
    # Load the model
    model = mujoco.load_model_from_path(model_path)

    # Print general model information
    print(f"Model name: {model.modelname}")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of degrees of freedom: {model.nv}")
    print(f"Number of position variables: {model.nq}")
    print()

    # Iterate through each joint and print information
    print("Joint Information:")
    for joint_id in range(model.njnt):
        joint_name = model.id2name(joint_id, mujoco.const.JNT)
        joint_type = model.jnt_type[joint_id]

        joint_type_str = {
            mujoco.const.JNT_FREE: "free",
            mujoco.const.JNT_BALL: "ball",
            mujoco.const.JNT_SLIDE: "slide",
            mujoco.const.JNT_HINGE: "hinge"
        }.get(joint_type, "unknown")

        # Collect position and velocity names based on joint type
        position_names = []
        velocity_names = []

        if joint_type == mujoco.const.JNT_FREE:
            position_names.extend([f"{joint_name}_quat_{i}" for i in range(4)])
            position_names.extend([f"{joint_name}_pos_{i}" for i in range(3)])
            velocity_names.extend([f"{joint_name}_angvel_{i}" for i in range(3)])
            velocity_names.extend([f"{joint_name}_vel_{i}" for i in range(3)])
        elif joint_type == mujoco.const.JNT_BALL:
            position_names.extend([f"{joint_name}_quat_{i}" for i in range(4)])
            velocity_names.extend([f"{joint_name}_angvel_{i}" for i in range(3)])
        elif joint_type == mujoco.const.JNT_SLIDE or joint_type == mujoco.const.JNT_HINGE:
            position_names.append(f"{joint_name}_pos")
            velocity_names.append(f"{joint_name}_vel")

        print(f"Joint ID: {joint_id}")
        print(f"  Name: {joint_name}")
        print(f"  Type: {joint_type_str}")
        print(f"  Position names: {position_names}")
        print(f"  Velocity names: {velocity_names}")
        print()

# Example usage
print_model_info(model_path)


model = mujoco.MjModel.from_xml_string(XML, ASSETS)
data = mujoco.MjData(model)
class Controller:
  def __init__(self,n_controls) -> None:
    self.P = 1
    self.I = 0.001
    self.D = -0.5
    self.prev_error = np.zeros(n_controls)
    self.integral = np.zeros(n_controls)
    
  def step_controler(self, states,reference):
    '''
    PID controller
    '''
    error = reference - states
    self.integral += error
    derivative = error - self.prev_error
    self.prev_error = error
    return self.P*error + self.I*self.integral + self.D*derivative

controller = Controller(model.nu)

def prephysic_step(model, data):
  states = get_states(model, data)
  control_output = controller.get_control_output(states)

def get_states(model, data):
  '''
  Get the states of the system by appending positions and velocities of the joints
  '''
  states = []
  for i in range(model.nq):
    states.append(data.qpos[i])
  for i in range(model.nv):
    states.append(data.qvel[i])
  return states

with mujoco.viewer.launch_passive(model,data) as viewer:
  start = time.time()

  while viewer.is_running() and time.time() - start < 500:
    step_start = time.time()
    
    prephysic_step(model,data)
    mujoco.mj_step(model, data)

    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
    
    viewer.sync()
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
