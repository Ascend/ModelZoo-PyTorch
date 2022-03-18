ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME
RUN git config --global http.sslverify false
COPY requirements.txt .
RUN pip3.7 install -r requirements.txt