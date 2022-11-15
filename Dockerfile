FROM balenalib/jetson-nano-ubuntu:bionic as final

WORKDIR /usr/src/app

# Don't prompt with any configuration questions
ENV DEBIAN_FRONTEND noninteractive

# Download and install BSP binaries for L4T 32.4.4
RUN apt-get update && apt-get install -y wget tar lbzip2 cuda-toolkit-10-2 cuda-compiler-10-2 libcudnn8 python3 python3-pip libegl1 mesa-common-dev libglu1-mesa-dev && \
    wget https://developer.nvidia.com/embedded/L4T/r32_Release_v4.4/r32_Release_v4.4-GMC3/T210/Tegra210_Linux_R32.4.4_aarch64.tbz2 && \
    tar xf Tegra210_Linux_R32.4.4_aarch64.tbz2 && \
    cd Linux_for_Tegra && \
    sed -i 's/config.tbz2\"/config.tbz2\" --exclude=etc\/hosts --exclude=etc\/hostname/g' apply_binaries.sh && \
    sed -i 's/install --owner=root --group=root \"${QEMU_BIN}\" \"${L4T_ROOTFS_DIR}\/usr\/bin\/\"/#install --owner=root --group=root \"${QEMU_BIN}\" \"${L4T_ROOTFS_DIR}\/usr\/bin\/\"/g' nv_tegra/nv-apply-debs.sh && \
    sed -i 's/LC_ALL=C chroot . mount -t proc none \/proc/ /g' nv_tegra/nv-apply-debs.sh && \
    sed -i 's/umount ${L4T_ROOTFS_DIR}\/proc/ /g' nv_tegra/nv-apply-debs.sh && \
    sed -i 's/chroot . \//  /g' nv_tegra/nv-apply-debs.sh && \
    ./apply_binaries.sh -r / --target-overlay && cd .. \
    rm -rf Tegra210_Linux_R32.4.4_aarch64.tbz2 && \
    rm -rf Linux_for_Tegra && \
    apt-get install -o DPkg::Options::="--force-confnew" -y nvidia-l4t-jetson-multimedia-api swig git &&  \
    echo "/usr/lib/aarch64-linux-gnu/tegra" > /etc/ld.so.conf.d/nvidia-tegra.conf && \
    echo "/usr/lib/aarch64-linux-gnu/tegra-egl" > /etc/ld.so.conf.d/nvidia-tegra-egl.conf && ldconfig

# Last line is required to get the nvargus-daemon working inside the container
# https://forums.developer.nvidia.com/t/nvargus-daemon-gstreamer-errors-libegl-mesa/144512/12

# Install argus camera driver
RUN apt update && apt install -y --fix-missing make g++ git cmake && \
    git clone -b awblock https://github.com/richardw347/argus_camera.git && \
    mkdir -p argus_camera/build && cd argus_camera/build && cmake ../ && \
    make -j 4 && make install && cd ../ && python3 setup.py install

# Install system depedencies
RUN \
    apt update && apt install -y --fix-missing make g++ && apt-get install -y apt-transport-https \
    libhdf5-serial-dev \
    hdf5-tools \
    pkg-config \
    libhdf5-100 \
    libhdf5-dev \
    zlib1g-dev \
    libjpeg8-dev \
    liblapack-dev \
    libblas-dev \
    gfortran \
    lbzip2  \
    xorg \
    libgit2-dev \
    avahi-daemon \
    dbus \
    libnss-mdns && \
    sed -i -r 's/^[[:blank:]]*#?[[:blank:]]*enable-dbus[[:blank:]]*=.*$/enable-dbus=no/' /etc/avahi/avahi-daemon.conf

# install onnxruntime
RUN pip3 install cython && \
    wget https://nvidia.box.com/shared/static/ukszbm1iklzymrt54mgxbzjfzunq7i9t.whl -O onnxruntime_gpu-1.7.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install onnxruntime_gpu-1.7.0-cp36-cp36m-linux_aarch64.whl

RUN ln -s /host/run/dbus /var/run/dbus

ENV UDEV=1

ENV LD_LIBRARY_PATH=/usr/local/lib

RUN mkdir -p /usr/src/app/jetson_detect
COPY . /usr/src/app/jetson_detect

WORKDIR /usr/src/app/jetson_detect
CMD [ "sleep", "infinity" ]
