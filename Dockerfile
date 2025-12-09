# Use a standard PyTorch/CUDA base image
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

#set environment variables
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Set python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

#Install required  Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#create app directory
WORKDIR /app
COPY . /app
