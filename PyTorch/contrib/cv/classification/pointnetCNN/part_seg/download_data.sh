#!/bin/bash

cur_path=$(dirname $(readlink -f $0))
shapenetcore_partanno_v0=`sed '/^shapenetcore_partanno_v0=/!d;s/.*=//' ${cur_path}/url.ini`
shapenet_part_seg_hdf5_data=`sed '/^shapenet_part_seg_hdf5_data=/!d;s/.*=//' ${cur_path}/url.ini`

# Download original ShapeNetPart dataset (around 1GB)
wget ${shapenetcore_partanno_v0}
unzip shapenetcore_partanno_v0.zip
rm shapenetcore_partanno_v0.zip

# Download HDF5 for ShapeNet Part segmentation (around 346MB)
wget ${shapenet_part_seg_hdf5_data}
unzip shapenet_part_seg_hdf5_data.zip
rm shapenet_part_seg_hdf5_data.zip

