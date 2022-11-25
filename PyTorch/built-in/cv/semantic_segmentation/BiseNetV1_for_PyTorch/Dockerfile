ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

RUN git config --global http.sslverify false

RUN apt update && apt install -y libsm6 libxext6 2to3 ffmpeg
RUN apt-get install -y libxrender-dev

RUN pip3.7 install pillow==9.3.0
RUN pip3.7 install torchvision==0.9.1