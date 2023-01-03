ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

RUN git config --global http.sslverify false

RUN chmod 777 /tmp
RUN apt update && apt install libsm6 libxext6 2to3 ffmpeg libxrender-dev unzip -y

RUN pip3.7 install pillow==9.3.0
RUN pip3.7 install torchvision==0.9.1
RUN pip3.7 install cityscapesscripts