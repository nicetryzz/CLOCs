FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN rm -rf /var/lib/apt/lists/* \
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        fish \
        openssl \
        libssl-dev \
        libc6-dev \ 
        gcc \
        zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm tk-dev \
        libsparsehash-dev libgl1-mesa-glx && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libboost-all-dev && \    
    ldconfig && \    
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN wget -qc https://tukaani.org/xz/xz-5.2.3.tar.gz && \
    tar zxf xz-5.2.3.tar.gz && \
    cd xz-5.2.3 && \
    ./configure --prefix /opt/xz && \
    make -j8 && \
    make install && \
    cd .. && \
    wget -qc https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz && \
    tar xf Python-3.6.5.tgz && \
    cd Python-3.6.5 && \
    ./configure --prefix=/usr/local/lib/python3.6 LDFLAGS="-L/opt/xz/lib" CPPFLAGS="-I/opt/xz/include" && \
    make -j8 && \
    make install -j8 && \
    ln -sf /usr/local/lib/python3.6/bin/python3.6 /usr/bin/python3.6 && \
    ln -sf /usr/local/lib/python3.6/bin/pip3.6 /usr/bin/pip3.6 && \
    ln -sf /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.6 /usr/local/bin/python && \
    rm -rf Python-3.6.5.tgz Python-3.6.5 xz-5.2.3.tar.gz xz-5.2.3  

RUN wget -qc https://cmake.org/files/v3.13/cmake-3.13.2-Linux-x86_64.tar.gz && \
    tar zxf cmake-3.13.2-Linux-x86_64.tar.gz && \
    mv cmake-3.13.2-Linux-x86_64 /opt/cmake-3.13.2 && \
    ln -sf /opt/cmake-3.13.2/bin/* /usr/bin/ && \
    rm cmake-3.13.2-Linux-x86_64.tar.gz

RUN python -m pip --no-cache-dir install pip -U && \
    python -m pip --no-cache-dir install --upgrade --trusted-host https://mirrors.huaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple \
        setuptools wheel scikit-image scipy numba pillow==6.2.2 pybind11 matplotlib fire tensorboardX protobuf opencv-python && \
    python -m pip --no-cache-dir install --upgrade --trusted-host https://mirrors.huaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple \
        numpy==1.16.1 jupyter ipython seaborn psutil flask shapely pyopengl && \
    python -m pip --no-cache-dir install --upgrade --trusted-host https://mirrors.huaweicloud.com -i https://mirrors.huaweicloud.com/repository/pypi/simple \
        torch==1.1.0 torchvision==0.3.0 

ADD rootfs /

WORKDIR /root

RUN cd spconv && \
    python setup.py bdist_wheel && \
    python -m pip install ./dist/spconv-1.0-cp36-cp36m-linux_x86_64.whl && \
    echo "export PYTHONPATH=$PYTHONPATH:/root/CLOCs/ " >> ~/.bashrc

ENV NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
ENV NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
ENV NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice

VOLUME [ "/root/dataset" ]
VOLUME [ "/root/data"]

EXPOSE 8888

CMD ["/bin/bash"]

# ======= make docker
# docker build -t clocs .
# docker run -it -v /home3/hqlab/chenqilong/dataset:/root/dataset clocs
# python create_data.py create_kitti_info_file /root/dataset/KITTI_DATASET/
# python create_data.py create_groundtruth_database /root/dataset/KITTI_DATASET/
# CUDA_VISIBLE_DEVICES=1 python ./pytorch/train.py train --config_path=./configs/car.fhd.config --model_dir=../second_model/
# CUDA_VISIBLE_DEVICES=1 python ./pytorch/train.py evaluate --config_path=./configs/car.fhd.config --model_dir=../second_model/ --measure_time=True --batch_size=1