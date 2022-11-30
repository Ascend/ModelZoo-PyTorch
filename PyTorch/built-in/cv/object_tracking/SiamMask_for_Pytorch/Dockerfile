ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

RUN chmod 777 /tmp
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libxrender-dev make make-guile wget jq -y

COPY requirements.txt .
RUN pip3.7 install -r requirements.txt