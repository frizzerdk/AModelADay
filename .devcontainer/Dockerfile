# Use an official NVIDIA runtime base image with CUDA
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set environment variable to avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Update and install essential packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    tmux \
    python-is-python3 \
    x11-apps \
    x11-xserver-utils \
    libxrender1 \
    libxtst6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure Git with user information
RUN git config --global user.name "frizzerdk" \
    && git config --global user.email "frizzerdk@gmail.com"

# [Optional] Install additional packages or perform other customizations
# RUN pip3 install some-package

# Set the working directory
WORKDIR /workspaces

# Copy requirements.txt to the container image
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Set environment variables
ENV PYTHONPATH="/workspaces/AModelADay:${PYTHONPATH}"
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Ensure python command points to python3
RUN echo "alias python=python3" >> /root/.bashrc
