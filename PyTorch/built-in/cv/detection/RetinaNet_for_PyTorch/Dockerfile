ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME
RUN mkdir ./retinanet_requirement
WORKDIR ./retinanet_requirement
COPY requirements.txt .
RUN mkdir requirements
COPY requirements/* ./requirements/
RUN pip3.7 install -r requirements.txt
RUN chmod 777 /tmp
RUN apt-get update && apt-get install -y git 2to3 libgl1-mesa-glx
RUN git config --global http.sslverify false
RUN mmcv=`sed '/^mmcv=/!d;s/.*=//' url.ini` && git clone -b v1.2.6 --depth=1 ${mmcv}
WORKDIR ./mmcv
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8
COPY mmcv_need/_functions.py ./mmcv/parallel/
COPY mmcv_need/distributed.py ./mmcv/parallel/
COPY mmcv_need/data_parallel.py ./mmcv/parallel/
COPY mmcv_need/builder.py ./mmcv/runner/optimizer/
COPY mmcv_need/dist_utils.py ./mmcv/runner/
COPY mmcv_need/optimizer.py ./mmcv/runner/hooks/
