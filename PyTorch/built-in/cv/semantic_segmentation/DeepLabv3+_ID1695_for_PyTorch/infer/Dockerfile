ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}

RUN ln -s  /usr/local/python3.7.5/bin/python3 /usr/bin/python

COPY sdk/requirements.txt .
RUN pip3.7 install -r requirements.txt
