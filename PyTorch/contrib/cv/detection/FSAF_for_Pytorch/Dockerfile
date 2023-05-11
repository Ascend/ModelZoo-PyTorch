ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME

RUN apt update && apt install -y git
RUN /usr/bin/python3 -m pip install --update pip 2to3 libgl1-mesa-glx
RUN pip install addict ninja

RUN mkdir ./fsaf_requirement
WORKDIR ./fsaf_requirement
COPY mmdetection/requirements.txt .
RUN mkdir requirements
COPY mmdetection/requirements/* ./requirements/
RUN pip3.7 install -r requirements.txt
RUN chmod 777 /tmp
RUN git config --global http.sslverify false
RUN git clone -b v1.2.7 https://github.com/open-mmlab/mmcv.git
WORKDIR ./mmcv
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8
RUN rm -rf mmcv
COPY mmcv_need ./mmcv
