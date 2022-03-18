#!/bin/bash

mkdir -p /opt/npu/datasets/citys/
ZIP_dir=/opt/npu/datasets/citys/
unzip ./gtFine_trainvaltest.zip -d $ZIP_dir
unzip ./leftImg8bit_trainvaltest.zip -d $ZIP_dir

mkdir -p ../datasets
ln -s /opt/npu/datasets/citys ../datasets/citys