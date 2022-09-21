ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME
RUN git config --global http.sslverify false
# os依赖
RUN chmod 777 /tmp && apt update && apt install -y git 2to3 libgl1-mesa-glx
# py依赖
RUN mkdir ./ssd_requirement
WORKDIR ./ssd_requirement
COPY requirements.txt .
RUN mkdir requirements
COPY requirements/* ./requirements/
RUN pip3.7 install cython
RUN pip3.7 install Pillow==9.1.0
RUN pip3.7 install -r requirements/build.txt
RUN pip3.7 install -r requirements/optional.txt
RUN pip3.7 install -r requirements/runtime.txt
# MMCV
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8
RUN git clone -b v1.4.8 --depth=1 https://github.com/open-mmlab/mmcv.git
WORKDIR ./mmcv
RUN pip3.7 install -r requirements/build.txt
RUN pip3.7 install -r requirements/optional.txt
RUN pip3.7 install -r requirements/runtime.txt
# torchvision
git clone -b v0.6.0 --depth=1 https://github.com/pytorch/vision.git
