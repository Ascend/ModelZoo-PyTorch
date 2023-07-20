#!/bin/bash

cur_path=$(dirname $(readlink -f $0))
indoor3d_sem_seg_hdf5_data=`sed '/^indoor3d_sem_seg_hdf5_data=/!d;s/.*=//' ${cur_path}/url.ini`

# Download HDF5 for indoor 3d semantic segmentation (around 1.6GB)
wget ${indoor3d_sem_seg_hdf5_data}
unzip indoor3d_sem_seg_hdf5_data.zip
rm indoor3d_sem_seg_hdf5_data.zip

