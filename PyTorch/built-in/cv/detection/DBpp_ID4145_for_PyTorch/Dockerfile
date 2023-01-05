ARG FROM_IMAGE_NAME=ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:22.0.RC3-1.8.1
FROM $FROM_IMAGE_NAME
COPY requirements.txt .
COPY requirements requirements
RUN pip3 install -r requirements.txt

# install official mmcv
RUN pip3 uninstall mmcv
RUN pip3 install mmcv-full -f http://download.openmmlab.com/mmcv/dist/npu/torch1.8.0/index.html

RUN echo "export LD_PRELOAD = $LD_PRELOAD:/usr/local/python3.7.5/lib/python3.7/site-packages/scikit-learn.libs/libgomp-d22c30c5.so.1.0.0"
