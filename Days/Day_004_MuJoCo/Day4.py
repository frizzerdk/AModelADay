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

class Controller:
  def __init__(self,n_controls,n_sensors,timestep=0.01) -> None:
    self.n_controls = n_controls
    self.n_sensors= n_sensors
    self.P = 0.1
    self.I = 0.00001
    self.D = 1
    self.prev_error = np.zeros(n_controls)
    self.integral = np.zeros(n_controls)
    self.timestep = timestep
    self.simulation_frequency= 1/timestep
    
  def step_controler(self, measurement ,B_matrix =None,reference = None):
    '''
    PID controller
    '''
    if B_matrix is None:
       B_matrix = np.eye(self.n_controls,self.n_sensors)
    if reference is None:
      reference = np.zeros(self.n_controls)
    error = reference - measurement
    self.integral += error*self.timestep
    derivative = (error - self.prev_error) * self.simulation_frequency
    self.prev_error = error
    P = self.P*error
    I = self.I*self.integral
    D = self.D*derivative
    return self.P*error + self.I*self.integral + self.D*derivative


def prephysic_step(model, data,controller):
  sensor_data = get_sensor_data(model, data)
  control_output = controller.step_controler(sensor_data)
  control_joint_map = [1]
  data.ctrl = control_output


def get_sensor_data(model, data):
  return data.qpos[0]

def set_initial_pose(data):
  # Set the initial position and velocity
  initial_qpos = np.zeros(data.qpos.shape)
  initial_qvel = np.zeros(data.qvel.shape)

  # Example: Set specific values for initial positions and velocities
  initial_qpos[0] = 0.0  # Example: initial position of the first DoF
  initial_qpos[1] = np.pi / 4  # Example: initial position of the second DoF (45 degrees)
  initial_qvel[0] = 0.0  # Example: initial velocity of the first DoF
  initial_qvel[1] = 0.0  # Example: initial velocity of the second DoF

  # Assign the initial pose to the data
  data.qpos[:] = initial_qpos
  data.qvel[:] = initial_qvel


def main():
  
  model = mujoco.MjModel.from_xml_string(XML, ASSETS)
  print_model_info(model)

  data = mujoco.MjData(model)
  n_controls = model.nu
  n_sensors = 1
  timestep= model.opt.timestep
  controller = Controller(n_controls,n_sensors,timestep=timestep)
  set_initial_pose(data)

  with mujoco.viewer.launch_passive(model,data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < 500:
      step_start = time.time()
      
      prephysic_step(model,data,controller)
      mujoco.mj_step(model, data)

      with viewer.lock():
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)
      
      viewer.sync()
      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = model.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)

def print_model_info(model):
    # Load the model

    # Print general model information
    print(f"Model information:")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of degrees of freedom: {model.nv}")
    print(f"Number of position variables: {model.nq}")
    print()

    # Iterate through each joint and print information
    print("Joint Information:")
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = model.jnt_type[joint_id]

        joint_type_str = {
            mujoco.mjtJoint.mjJNT_FREE: "free",
            mujoco.mjtJoint.mjJNT_BALL: "ball",
            mujoco.mjtJoint.mjJNT_SLIDE: "slide",
            mujoco.mjtJoint.mjJNT_HINGE: "hinge"
        }.get(joint_type, "unknown")

        # Collect position and velocity names based on joint type
        position_names = []
        velocity_names = []

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            position_names.extend([f"{joint_name}_quat_{i}" for i in range(4)])
            position_names.extend([f"{joint_name}_pos_{i}" for i in range(3)])
            velocity_names.extend([f"{joint_name}_angvel_{i}" for i in range(3)])
            velocity_names.extend([f"{joint_name}_vel_{i}" for i in range(3)])
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            position_names.extend([f"{joint_name}_quat_{i}" for i in range(4)])
            velocity_names.extend([f"{joint_name}_angvel_{i}" for i in range(3)])
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE or joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            position_names.append(f"{joint_name}_pos")
            velocity_names.append(f"{joint_name}_vel")

        print(f"Joint ID: {joint_id}")
        print(f"  Name: {joint_name}")
        print(f"  Type: {joint_type_str}")
        print(f"  Position names: {position_names}")
        print(f"  Velocity names: {velocity_names}")
        print()
    # Iterate through each actuator and print information
    print("Actuator Information:")
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        actuator_trntype = model.actuator_trntype[actuator_id]

        actuator_trntype_str = {
            mujoco.mjtTrn.mjTRN_JOINT: "joint",
            mujoco.mjtTrn.mjTRN_SLIDERCRANK: "slidercrank",
            mujoco.mjtTrn.mjTRN_TENDON: "tendon",
            mujoco.mjtTrn.mjTRN_SITE: "site",
            mujoco.mjtTrn.mjTRN_BODY: "body"
        }.get(actuator_trntype, "unknown")

        print(f"Actuator ID: {actuator_id}")
        print(f"  Name: {actuator_name}")
        print(f"  Transmission type: {actuator_trntype_str}")
        print()

if __name__ == '__main__':  
   main()