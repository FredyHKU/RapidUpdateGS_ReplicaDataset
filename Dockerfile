FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1-mesa \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    mesa-utils \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH and accept Terms of Service
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_ACCEPT_TOS=yes

# Initialize conda for bash
RUN conda init bash && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment for habitat
RUN CONDA_ACCEPT_TOS=yes conda create -n habitat python=3.9 cmake=3.14.0 -y

# Activate environment and install habitat-sim
# Using full version for better compatibility without GPU
RUN /bin/bash -c "source activate habitat && \
    CONDA_ACCEPT_TOS=yes conda install habitat-sim withbullet -c conda-forge -c aihabitat -y"

# Clone habitat-sim repository for examples and demos
RUN git clone https://github.com/facebookresearch/habitat-sim.git /habitat-sim

# Set working directory
WORKDIR /workspace

# Configure environment variables for X11
ENV DISPLAY=host.docker.internal:0.0
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_INDIRECT=0

# Make sure conda environment is activated by default
RUN echo "source activate habitat" >> ~/.bashrc

# Install example code / replica dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    pigz \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN /bin/bash -c "source activate habitat && \
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /habitat-sim/test_data"

RUN /bin/bash -c "source activate habitat && \  
    python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path /habitat-sim/test_data"

# Install recorder and generator dependencies
COPY workspace/requirements.txt /tmp/requirements.txt
RUN /bin/bash -c "source activate habitat && \
    pip install -r /tmp/requirements.txt"

# Set the default shell to bash
SHELL ["/bin/bash", "-c"]

# Default command
CMD ["/bin/bash"]
