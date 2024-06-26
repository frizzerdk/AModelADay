import os
# change dir to dir of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import mujoco
import mujoco.viewer
import time
import numpy as np
import math
import asyncio
import websockets
import json
import time
model_path = 'quadbot.xml'
# load XML from file
with open(model_path, 'r') as f:
  XML = f.read()

ASSETS=dict()
# with open('body.stl', 'rb') as f:
#   ASSETS['body.stl'] = f.read()
'''
  <actuator>
      <motor name="motor_FL_drive" joint="drive_FL_joint"/>
      <motor name="motor_FR_drive" joint="drive_FR_joint"/>
      <motor name="motor_BL_drive" joint="drive_BL_joint"/>
      <motor name="motor_BR_drive" joint="drive_BR_joint"/>
      <motor name="motor_FL_steer" joint="steer_FL_joint"/>
      <motor name="motor_FR_steer" joint="steer_FR_joint"/>
      <motor name="motor_BL_steer" joint="steer_BL_joint"/>
      <motor name="motor_BR_steer" joint="steer_BR_joint"/>
    </actuator>
    <sensor>
      <jointpos name="steerpos_FL" joint="steer_FL_joint"/>
      <jointpos name="steerpos_FR" joint="steer_FR_joint"/>
      <jointpos name="steerpos_BL" joint="steer_BL_joint"/>
      <jointpos name="steerpos_BR" joint="steer_BR_joint"/>
      <jointvel name="drivevel_FL" joint="drive_FL_joint"/>
      <jointvel name="drivevel_FR" joint="drive_FR_joint"/>
      <jointvel name="drivevel_BL" joint="drive_BL_joint"/>
      <jointvel name="drivevel_BR" joint="drive_BR_joint"/>
    </sensor> 
'''
############################################
# Main Loop
############################################

def prephysic_step(model, data,controller,run_controller=True):
  sensor_data = get_all_sensor_data(data)
  mapped_sensor_data = map_sensor_data(sensor_data)
  if run_controller:
    control_output= controller.step_controler(measurement=mapped_sensor_data)
    mapped_control_output =map_control_output(control_output)
    apply_control(model,data,mapped_control_output)

############################################
# Controller
############################################
# PID Controllers

class PIDController:
    def __init__(self, n_controls=1, timestep=0.01, kP=1, kI=1, kD=0.01, outputScale=0.1):
        self.n_controls = n_controls
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.P = 0
        self.I = 0
        self.D = 0
        self.prev_error = np.zeros(n_controls)
        self.integral = np.zeros(n_controls)
        self.timestep = timestep
        self.simulation_frequency = 1 / timestep
        self.outputScale = outputScale
        self.lastOutput = 0

    def step_controller(self, measurement, reference=None):
        if reference is None:
            reference = np.zeros(self.n_controls)
        error = reference - measurement
        self.integral += error * self.timestep
        derivative = (error - self.prev_error) * self.simulation_frequency
        self.prev_error = error
        self.P = self.kP * error
        self.I = self.kI * self.integral
        self.D = self.kD * derivative
        self.lastOutput = (self.P + self.I + self.D) * self.outputScale
        return self.lastOutput

    def to_json(self):
        json_data = {
            "kP": self.kP,
            "kI": self.kI,
            "kD": self.kD,
            "outputScale": self.outputScale,
            "prev_error": self.prev_error.tolist() if isinstance(self.prev_error, np.ndarray) else self.prev_error,
            "integral": self.integral.tolist() if isinstance(self.integral, np.ndarray) else self.integral,
            "P": self.P.tolist() if isinstance(self.P, np.ndarray) else self.P,
            "I": self.I.tolist() if isinstance(self.I, np.ndarray) else self.I,
            "D": self.D.tolist() if isinstance(self.D, np.ndarray) else self.D,
            "lastOutput": self.lastOutput.tolist() if isinstance(self.lastOutput, np.ndarray) else self.lastOutput
        }
        return json_data


class PIDControllerVectorAngle(PIDController):
    def __init__(self, n_controls=1, timestep=0.01, kP=2, kI=0.5, kD=0.1, outputScale=1):
        super().__init__(n_controls, timestep, kP, kI, kD, outputScale)
   
    def angle_to_vector(self, angle):
        #rad = math.radians(angle)
        rad = angle
        return math.cos(rad), math.sin(rad)

    def vectors_to_angle(self, v1, v2, in_degrees=True):
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        dot_product = max(min(dot_product, 1.0), -1.0)
        angle = math.acos(dot_product)
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        if cross_product < 0:
            angle = -angle
        return math.degrees(angle) if in_degrees else angle

    def is_opposite_vector(self, vec1, vec2):
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        return dot_product < -0.999999

    def vector_error(self, desired_angle, current_angle):
        desired_vector = self.angle_to_vector(desired_angle)
        current_vector = self.angle_to_vector(current_angle)
        if self.is_opposite_vector(desired_vector, current_vector):
            return 180.0
        error_angle = self.vectors_to_angle(desired_vector, current_vector)
        if error_angle > 180:
            error_angle -= 360
        elif error_angle < -180:
            error_angle += 360
        return -error_angle

    def step_controller(self, measurement_angle, reference_angle=None):
        if reference_angle is None:
            reference_angle = 0
        error = self.vector_error(reference_angle, measurement_angle) / 360
        self.integral += error * self.timestep
        derivative = (error - self.prev_error) * self.simulation_frequency
        self.prev_error = error
        self.P = self.kP * error
        self.I = self.kI * self.integral
        self.D = self.kD * derivative
        self.lastOutput = (self.P + self.I + self.D) * self.outputScale
        return self.lastOutput

  

class QuadController:
  ''' Controller for the quadbot
      Actuators:
        - drive: FL, FR, BL, BR
        - steer: FL, FR, BL, BR
      Sensors:
        - drive vel: FL, FR, BL, BR
        - steer pos: FL, FR, BL, BR
      Reference:
        - Desired Instantaneous Circle of Rotation ( mapped for x)
        - Desired Velocity average velocity of the wheels

  '''
  def __init__(self,width = 1,length = 1) -> None:
    self.measurement = {'drive_FL':0,'drive_FL':0,'drive_BL':0,'drive_BR':0,
                        'steer_FL':0,'steer_FR':0,'steer_BL':0,'steer_BR':0}
        
    self.LocalControllers = {'drive_FL':PIDController(),
                             'drive_FR':PIDController(),
                             'drive_BL':PIDController(),
                             'drive_BR':PIDController(),
                             'steer_FL':PIDControllerVectorAngle(),
                             'steer_FR':PIDControllerVectorAngle(),
                             'steer_BL':PIDControllerVectorAngle(),
                             'steer_BR':PIDControllerVectorAngle()}
    
    self.targets = {'drive_FL':0,'drive_FR':0,'drive_BL':0,'drive_BR':0,
                    'steer_FL':0,'steer_FR':0,'steer_BL':0,'steer_BR':0}
   
    self.control_reference = {'velocity':0,'ICR_x':0.1,'ICR_y':1}
    
    self.width = width
    self.length = length
    self.wheel_names = ['FL','FR','BL','BR']
    self.wheel_positions = [[-self.width/2,self.length/2],
                      [self.width/2,self.length/2],
                      [-self.width/2,-self.length/2],
                      [self.width/2,-self.length/2]]
    self.wheel_pos_dict = {self.wheel_names[i]:self.wheel_positions[i] for i in range(4)}
    

    
  def steering_angle(self,ICR_x,ICR_y,point_x,point_y):
    # Calculate the steering angle from the ICR
    angle = np.arctan2(point_y-ICR_y,point_x-ICR_x)
    return angle if ICR_x>0 else angle-np.pi

  def wheel_speed(self,ICR_x,ICR_y,point_x,point_y, target_speed=1):
      # Calculate the wheel speed
      # Essensially the speed is just proportional to the distance to ICR from wheel
      if ICR_x == np.inf or ICR_y == np.inf:
        return np.ones(4)*target_speed
      
      speed = np.sqrt((point_x-ICR_x)**2 + (point_y-ICR_y)**2)
      #print("relative speed",speed)
      speed = speed / np.mean(speed)
      #print("out speed",speed)
      #print("sum",np.sum(speed))
      return speed*target_speed
  

  def steering_map(self,new):
    # clip input to range [-1,1]
    eps=0.00001
    x=np.clip(new+eps,-1,1)
    new_map = np.sign(x) * (1 / abs(x) - 1) if x != 0 else np.inf
    return new_map
  
  def set_reference(self,velocity=None,ICR_x=None,ICR_y=None,Delta=False):
    if Delta:
      self.control_reference['velocity'] += velocity if velocity else 0
      self.control_reference['ICR_x'] += ICR_x if ICR_x else 0
      self.control_reference['ICR_y'] += ICR_y if ICR_y else 0
    else:
      self.control_reference['velocity'] = velocity if velocity else self.control_reference['velocity']
      self.control_reference['ICR_x'] = ICR_x if ICR_x else self.control_reference['ICR_x']
      self.control_reference['ICR_y'] = ICR_y if ICR_y else self.control_reference['ICR_y']
    return self.control_reference

  def step_controler(self,measurement=None,reference = None):
    self.measurement = measurement if measurement else self.measurement
    # get reference speed and angles
    self.control_reference = reference if reference else self.control_reference

    # Calculate the targets for  all wheels at once to get adjusted speed and angles
    self.angles = self.steering_angle(self.steering_map(self.control_reference['ICR_x']),self.steering_map(self.control_reference['ICR_y']),
                                      np.array([x for x, y in self.wheel_pos_dict.values()]),
                                      np.array([y for x, y in self.wheel_pos_dict.values()]))
    self.speeds = self.wheel_speed(self.steering_map(self.control_reference['ICR_x']),self.steering_map(self.control_reference['ICR_y']),
                                      np.array([x for x, y in self.wheel_pos_dict.values()]),
                                      np.array([y for x, y in self.wheel_pos_dict.values()]),
                                      self.control_reference['velocity'])
    outputs = {}
    # run PID for each and make output dict
    for index,  wheel in enumerate(self.wheel_names):
      self.targets[f"drive_{wheel}"] = self.speeds[index]
      self.targets[f"steer_{wheel}"] = self.angles[index]
      outputs[f"drive_{wheel}"] = self.LocalControllers[f"drive_{wheel}"].step_controller(self.measurement[f"drive_{wheel}"],self.targets[f"drive_{wheel}"])
      outputs[f"steer_{wheel}"] = self.LocalControllers[f"steer_{wheel}"].step_controller(self.measurement[f"steer_{wheel}"],self.targets[f"steer_{wheel}"])
    
    return outputs
  
  def to_json(self):
     # Add all data to a json struct
    json_data = { 
        "controllers" : {
            "drive_FL" : self.LocalControllers['drive_FL'].to_json(),
            "drive_FR" : self.LocalControllers['drive_FR'].to_json(),
            "drive_BL" : self.LocalControllers['drive_BL'].to_json(),
            "drive_BR" : self.LocalControllers['drive_BR'].to_json(),
            "steer_FL" : self.LocalControllers['steer_FL'].to_json(),
            "steer_FR" : self.LocalControllers['steer_FR'].to_json(),
            "steer_BL" : self.LocalControllers['steer_BL'].to_json(),
            "steer_BR" : self.LocalControllers['steer_BR'].to_json()
        },
        "targets" : self.targets,
        "control_reference" : self.control_reference

    }
    return json_data
############################################
# Util Functions
############################################


class WebSocketClient:
    def __init__(self, uri="ws://localhost:9871"):
        self.uri = uri
        self.websocket = None
        self.loop = asyncio.get_event_loop()

    async def connect(self):
        self.websocket = await websockets.connect(self.uri)

    async def send_json_data(self, data):
        if self.websocket is not None:
            await self.websocket.send(json.dumps(data))
            #print(f"Sent: {data}")

    async def close(self):
        if self.websocket is not None:
            await self.websocket.close()

    def send_data(self, data):
        # This method can be called from other parts of the code
        self.loop.run_until_complete(self.send_json_data(data))

    def start_connection(self):
        # Start the connection
        self.loop.run_until_complete(self.connect())

    def close_connection(self):
        # Close the connection
        self.loop.run_until_complete(self.close())

def map_sensor_data(sensor_data):
  map = {'drivevel_FL':'drive_FL','drivevel_FR':'drive_FR',
          'drivevel_BL':'drive_BL','drivevel_BR':'drive_BR',
          'steerpos_FL':'steer_FL','steerpos_FR':'steer_FR',
          'steerpos_BL':'steer_BL','steerpos_BR':'steer_BR'}
  return {map[key]:sensor_data[key] for key in sensor_data.keys()}


def get_sensor_data_by_name(data, sensor_name):
    sensor_id = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    sensor_start = data.model.sensor_adr[sensor_id]
    sensor_dim = data.model.sensor_dim[sensor_id]
    sensor_data = data.sensordata[sensor_start:sensor_start + sensor_dim]
    return sensor_data


def get_all_sensor_data(data):
    sensor_names = [
        'drivevel_FL', 'drivevel_FR', 'drivevel_BL', 'drivevel_BR',
        'steerpos_FL', 'steerpos_FR', 'steerpos_BL', 'steerpos_BR'
    ]
    sensor_data = {name: get_sensor_data_by_name(data, name) for name in sensor_names}
    return sensor_data

def map_control_output(control_output):
  map = {'drive_FL':'motor_FL_drive','drive_FR':'motor_FR_drive',
          'drive_BL':'motor_BL_drive','drive_BR':'motor_BR_drive',
          'steer_FL':'motor_FL_steer','steer_FR':'motor_FR_steer',
          'steer_BL':'motor_BL_steer','steer_BR':'motor_BR_steer'}
  return {map[key]:control_output[key] for key in control_output.keys()}

def apply_control(model, data, control_output):
    # Set the control output for each actuator
    for actuator_name, output in control_output.items():
        set_actuator_output(model, data, actuator_name, output)

def set_actuator_output(model, data, actuator_name, output):
    """
    Set the output of an actuator based on its name.

    :param model: The MuJoCo model object.
    :param data: The MuJoCo data object.
    :param actuator_name: The name of the actuator.
    :param output: The desired output value for the actuator.
    """
    # Find the actuator ID based on the name
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    
    if actuator_id == -1:
        raise ValueError(f"Actuator '{actuator_name}' not found in the model.")
    
    # Set the control output for the actuator
    data.ctrl[actuator_id] = output[:]

def load_model(XML, ASSETS):
  model = mujoco.MjModel.from_xml_string(XML, ASSETS)
  data = mujoco.MjData(model)
  return model, data



def get_joint_position_by_name(model, data, joint_name):
    # Get the joint ID from the joint name
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    
    # Get the starting index of the joint's position in qpos
    qpos_start = model.jnt_qposadr[joint_id]
    
    # Determine the number of qpos entries for this joint based on joint type
    joint_type = model.jnt_type[joint_id]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        qpos_length = 7  # Free joints have 7 qpos entries (4 for quaternion + 3 for position)
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        qpos_length = 4  # Ball joints have 4 qpos entries (quaternion)
    else:
        qpos_length = 1  # Slide and hinge joints have 1 qpos entry

    # Get the joint position(s)
    joint_position = data.qpos[qpos_start:qpos_start + qpos_length]

    return joint_position

def set_initial_pose(data):
  # Set the initial position and velocity
  # initial_qpos = np.zeros(data.qpos.shape)
  initial_qvel = np.zeros(data.qvel.shape)

  # # Assign the initial pose to the data
  # data.qpos[:] = initial_qpos
  data.qvel[:] = initial_qvel

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

        # Collect position and velocity nrogileames based on joint type
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

def get_joint_name(model, joint_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

def get_actuator_name(model, actuator_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)

def get_sensor_name(model, sensor_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)

def mujoco_data_to_dict(model, data):
    joint_data = {}
    for joint_id in range(model.njnt):
        joint_name = get_joint_name(model, joint_id)
        joint_type = model.jnt_type[joint_id]
        
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            joint_data[joint_name] = {
                "qpos": data.qpos[model.jnt_qposadr[joint_id]:model.jnt_qposadr[joint_id] + 7].tolist(),
                "qvel": data.qvel[model.jnt_dofadr[joint_id]:model.jnt_dofadr[joint_id] + 6].tolist()
            }
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            joint_data[joint_name] = {
                "qpos": data.qpos[model.jnt_qposadr[joint_id]:model.jnt_qposadr[joint_id] + 4].tolist(),
                "qvel": data.qvel[model.jnt_dofadr[joint_id]:model.jnt_dofadr[joint_id] + 3].tolist()
            }
        elif joint_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            joint_data[joint_name] = {
                "qpos": data.qpos[model.jnt_qposadr[joint_id]],
                "qvel": data.qvel[model.jnt_dofadr[joint_id]]
            }
    
    actuator_data = {}
    for actuator_id in range(model.nu):
        actuator_name = get_actuator_name(model, actuator_id)
        actuator_data[actuator_name] = data.ctrl[actuator_id]

    sensor_data = {}
    for sensor_id in range(model.nsensor):
        sensor_name = get_sensor_name(model, sensor_id)
        sensor_data[sensor_name] = data.sensordata[model.sensor_adr[sensor_id]:model.sensor_adr[sensor_id] + model.sensor_dim[sensor_id]].tolist()

    data_dict = {
        "joints": joint_data,
        "actuators": actuator_data,
        "sensors": sensor_data,
        "time": data.time
    }
    
    return data_dict





def main():
  webclient=WebSocketClient()
  webclient.start_connection()
  ###### Initialization ######
  model, data = load_model(XML, ASSETS)
  controller = QuadController(1,1)
  set_initial_pose(data)
  print_model_info(model)
  ############################
    # Define the key event handler
  def key_callback(keycode):

      key = chr(keycode)
      if key == chr( 265): # up 
        controller.set_reference(velocity=5,Delta=True)
      elif key == chr( 264): # down
        controller.set_reference(velocity=-5,Delta=True)
      elif key == chr( 263): # left
        controller.set_reference(ICR_x=0.1,Delta=True)
      elif key == chr( 262): # right
        controller.set_reference(ICR_x=-0.1,Delta=True)
      elif key == '.':
        controller.set_reference(ICR_y=0.1,Delta=True)
      elif key == ',':
        controller.set_reference(ICR_y=-0.1,Delta=True)
      print(controller.control_reference)
        

  with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    start = time.time()
    n_steps=0
    while viewer.is_running() and time.time() - start < 1000:
      step_start = time.time()
      n_steps+=1
      ###### loop ######
      run_controller =  n_steps%10
      prephysic_step(model,data,controller,run_controller)
      mujoco.mj_step(model, data)
      ##################
      jsondata = mujoco_data_to_dict(model, data)
      jsondata["controller"] = controller.to_json()
      webclient.send_data(jsondata)
      
      viewer.sync()
      # Rudimentary time keeping, will drift relative to wall clock.
      time_until_next_step = model.opt.timestep - (time.time() - step_start)
      if time_until_next_step > 0:
        time.sleep(time_until_next_step)
if __name__ == '__main__':  
   main()