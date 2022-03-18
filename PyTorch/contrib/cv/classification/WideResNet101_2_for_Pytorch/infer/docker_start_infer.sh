#!/bin/bash

docker_image=$1
model_dir=$2

docker run -it --ipc=host \
	      --device=/dev/davinci0 \
              --device=/dev/davinci_manager \
              --device=/dev/devmm_svm --device=/dev/hisi_hdc \
              -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
              -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
              -v ${model_dir}:${model_dir} \
              -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
              -v /var/log/npu/slog/:/var/log/npu/slog -v /var/log/npu/profiling/:/var/log/npu/profiling \
              -v /var/log/npu/dump/:/var/log/npu/dump -v /var/log/npu/:/usr/slog ${docker_image} \
              /bin/bash
