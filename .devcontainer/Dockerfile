# Use an official NVIDIA runtime base image with CUDA
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04


RUN apt-get update && apt-get install -y pip
# Install Git
RUN apt-get update && apt-get install -y git

RUN git config --global user.name "frizzerdk"
RUN git config --global user.email "frizzerdk@gmail.com"


# [Optional] Install additional packages or perform other customizations
# RUN pip3 install some-package

# Set the working directory
WORKDIR /workspace
# Copy requirements.txt to the container image

COPY requirements.txt .


# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt
ENV PYTHONPATH="/workspaces/Kaggle:${PYTHONPATH}"
ENV PATH /usr/local/cuda/bin:${PATH}
