ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME
RUN mkdir requirements
COPY requirements/* ./requirements/
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt
RUN apt-get update && apt-get install -y git 2to3 libgl1-mesa-glx
RUN git config --global http.sslverify false
ENV MMCV_WITH_OPS=1
ENV MAX_JOBS=8