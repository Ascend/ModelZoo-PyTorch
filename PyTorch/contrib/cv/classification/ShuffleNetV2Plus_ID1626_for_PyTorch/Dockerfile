ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME

COPY requirements.txt .
RUN aliyun=`sed '/^aliyun=/!d;s/.*=//' url.ini` && pip3.7 install -r requirements.txt -i ${aliyun} --trusted-host mirrors.aliyun.com