ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME
RUN mkdir ./deeplabv3_requirement
WORKDIR ./deeplabv3_requirement
COPY requirements.txt .
RUN mkdir requirements
COPY requirements/* ./requirements/
RUN pip3.7 install -r requirements.txt
RUN apt-get update && apt-get install -y git 2to3 libgl1-mesa-glx
RUN git config --global http.sslverify false
RUN git clone -b v1.3.9 --depth=1 https://github.com/open-mmlab/mmcv.git
WORKDIR ./mmcv
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8
RUN python3 setup.py build_ext
RUN python3 setup.py develop
RUN pip3.7 uninstall opencv-python
RUN pip3.7 install opencv-python-headless
COPY mmcv_need/_functions.py ./mmcv/parallel/
COPY mmcv_need/scatter_gather.py ./mmcv/parallel/
COPY mmcv_need/dist_utils.py ./mmcv/runner/
