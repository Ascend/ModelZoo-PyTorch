ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME

COPY xtcocoapi.zip /tmp/
RUN unzip /tmp/xtcocoapi.zip
WORKDIR /tmp/xtcocoapi
RUN python3 setup.py build_ext install

WORKDIR /tmp/
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt
RUN apt-get update && apt-get install -y git 2to3 libgl1-mesa-glx