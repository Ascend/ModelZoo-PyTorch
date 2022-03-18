FROM swr.cn-south-1.myhuaweicloud.com/public-ascendhub/ascend-pytorch-x86:21.0.1

COPY ./requirements.txt ./
RUN pip3.7 install -r requirements.txt
RUN chmod 777 /tmp
RUN apt update 
RUN apt install -y libglvnd-dev