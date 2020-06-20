# ==================================================================
# module list
# ------------------------------------------------------------------
# 
# ==================================================================

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV PATH /opt/conda/bin:$PATH

# ==================================================================
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    APT_INSTALL="apt-get install -y --no-install-recommends" && \
    GIT_CLONE="git clone" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple" && \
    CONDA_INSTALL="conda install -y" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        apt-utils \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        htop \
        tmux \
        openssh-client \
        openssh-server \
        libboost-dev \
        libboost-thread-dev \
        libboost-filesystem-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        && \
# ==================================================================
# miniconda
# ------------------------------------------------------------------
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda && \
        rm ~/miniconda.sh && \
# ==================================================================
# conda
# ------------------------------------------------------------------
    conda config --set show_channel_urls yes && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    $CONDA_INSTALL \
        python=3.7 && \
    $CONDA_INSTALL \
        ipython \
        Cython \
        ffmpeg \
        sk-video \
	scikit-image \
        imageio \
        h5py \
        tensorboardx \
        torchvision \
        cffi \
        cudatoolkit=10.1 \
        pytorch=1.4 \
        torchvision \
        pandas \
        && \
    $PIP_INSTALL \
        lmdb \
        opencv-contrib-python \
        matplotlib \
        tqdm \
        scikit-learn \
        pytest \
        wget \
        gym \
        && \
    conda clean -y --all && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# ==================================================================
# # ctcdecode
# # ------------------------------------------------------------------
RUN git clone --recursive https://github.com/Jevin754/ctcdecode.git /usr/local/ctcdecode && \
    cd /usr/local/ctcdecode && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple .

# ==================================================================
# update gcc g++ to version 6.5, SCTK toolkit
# ------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y iputils-ping software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y gcc-6 g++-6 && \
    rm /usr/bin/gcc /usr/bin/g++ && \
    ln -s /usr/bin/gcc-6 /usr/bin/gcc && ln -s /usr/bin/g++-6 /usr/bin/g++ && \
    git clone --recursive https://github.com/usnistgov/SCTK.git /usr/local/SCTK && \
    cd /usr/local/SCTK && \
    make config && make all && make check && make install && make doc && \
    cp /usr/local/SCTK/bin/sclite /usr/bin

# ==================================================================
# install additional python package
# ------------------------------------------------------------------
RUN echo "export CPATH=/usr/local/cuda/include:$CPATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc && \
    pip install --verbose --no-cache-dir torch-scatter && \
    pip install --verbose --no-cache-dir torch-sparse && \
    pip install --verbose --no-cache-dir torch-cluster && \
    pip install --verbose --no-cache-dir torch-spline-conv && \
    pip install torch-geometric
