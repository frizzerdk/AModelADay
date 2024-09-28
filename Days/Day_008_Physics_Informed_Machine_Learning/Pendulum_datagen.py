import os
import random   

folder_out = "data/pendulum"
os.makedirs(folder_out, exist_ok=True)  
class Config_pendulum:

    def __init__(self):
        
        self.config_range = {
            "mass": [0.1, 10],
            "length": [0.1, 10],
            "damping": [0.1, 10],
            "friction": [0.1, 10],
            "gravity": [0.1, 10],
            "framerate": [5, 240],
            "camera_distance": [1,10],
            "camera_height": [1,10],
            "camera_zoom": [1,10],
            "camera_pitch": [-10,10],
            "camera_yaw": [-10,10],
            "camera_roll": [-10,10],
            "focal_length": [1,10],
            "pendulum_start_angle": [-10,10],
            "pendulum_start_angular_velocity": [-10,10],
            "time_sim": 10
            }
        self.config = self.generate_config()
        self.pendulum_xml = self.generate_pendulum_xml()
    def generate_config(self):
        config = {}
        for key, value in self.config_range.items():
            if isinstance(value, list) and len(value) == 2:
                if all(isinstance(i, int) for i in value):
                    config[key] = random.randint(value[0], value[1])
                elif all(isinstance(i, float) for i in value):
                    config[key] = random.uniform(value[0], value[1])
                else:
                    raise ValueError(f"Range for {key} must be all integers or all floats.")
            else:
                config[key] = value
        return config
    def generate_pendulum_xml(self):
        self.pendulum_xml = f"""
        <pendulum>
            <mass>{self.config['mass']}</mass>
            <length>{self.config['length']}</length>
            <damping>{self.config['damping']}</damping>
            <friction>{self.config['friction']}</friction>
            <gravity>{self.config['gravity']}</gravity>
        </pendulum>
        """
        return pendulum_xml
    def get_value(self, key):
        return self.config[key]
    def get_config(self):
        return self.config
    def get_config_range(self):
        return self.config_range
    def save_config(self):
        with open(os.path.join(folder_out, "config.txt"), "w") as f:
            for key, value in self.config.items():
                f.write(f"{key}: {value}\n")
    def save_config_range(self):
        with open(os.path.join(folder_out, "config_range.txt"), "w") as f:
            for key, value in self.config_range.items():
                f.write(f"{key}: {value}\n")
    def load_config(self):
        with open(os.path.join(folder_out, "config.txt"), "r") as f:
            for line in f:
                key, value = line.split(":")
                self.config[key] = value
    def load_config_range(self):
        with open(os.path.join(folder_out, "config_range.txt"), "r") as f:
            for line in f:
                key, value = line.split(":")
                self.config_range[key] = value


def sim(config):
    

N_Sims = 10

for i in range(N_Sims):
    config = Config_pendulum()
    sim(config)
    