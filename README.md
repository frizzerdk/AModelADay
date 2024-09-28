# AModelADay
Train Any ML Model every day


# install steps
* Install docker
* get env file
* Get nvidia driver
** make sure they are up to date
*** nvidia-smi
*** sudo apt-get purge nvidia*
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    sudo apt-get install nvidia-driver-560  # or the latest version number
* get cuda toolkit 
* get container toolkit
* install kaggle
* run xhost +local:docker
* install plotjuggler



sudo apt-get purge nvidia*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install nvidia-driver-560  # or the latest version number


# Apendix A
To install the latest CUDA driver, you'll want to follow these steps:

1. First, check your current NVIDIA driver version:
   ```
   nvidia-smi
   ```

2. Remove any existing NVIDIA drivers (optional, but recommended to avoid conflicts):
   ```
   sudo apt-get purge nvidia*
   ```

3. Add the NVIDIA repository:
   ```
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt-get update
   ```

4. Install the latest NVIDIA driver:
   ```
   sudo apt-get install nvidia-driver-535  # or the latest version number
   ```
   You can replace 535 with the latest version available.

5. To install CUDA toolkit (which includes the driver):
   
   a. Go to NVIDIA's CUDA download page: https://developer.nvidia.com/cuda-downloads
   
   b. Select your operating system and version.
   
   c. Follow the instructions provided there. It will be something like: (!! update !!)

   ```
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu....!!
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/.....!!
   sudo dpkg -i cuda-repo-ubuntu....!!
   sudo cp /var/cuda-repo-ubuntu...!!
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

6. After installation, reboot your system:
   ```
   sudo reboot
   ```

7. After rebooting, verify the installation:
   ```
   nvidia-smi
   ```
   and
   ```
   nvcc --version
   ```

Remember to adjust your PATH to include CUDA binaries:

```
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

After this, your system should have the latest CUDA driver and toolkit installed. Make sure to rebuild your Docker container after updating the driver to ensure compatibility.