#!/bin/bash

data_dir=$1
curr_dir=$PWD

mkdir -p $data_dir

cd $data_dir
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -O - | tar -xz

cd $curr_dir