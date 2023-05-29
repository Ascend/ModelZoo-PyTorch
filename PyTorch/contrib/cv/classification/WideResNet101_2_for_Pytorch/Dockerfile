ARG FROM_IMAGE_NAME
FROM $FROM_IMAGE_NAME


COPY requirements.txt .
#RUN sudo apt-get install cmake
#RUN pip3.7 install protobuf 
#RUN sudo apt-get install protobuf-compiler libprotoc-dev
# RUN /usr/bin/python3 -m pip install --upgrade pip 
RUN pip3.7 install -r requirements.txt
# RUN git clone https://github.com/NVIDIA/apex
# RUN cd apex
# RUN pip install -v --no-cache-dir --global-options="cpp_ext" --global-option="--cuda_ext" ./
