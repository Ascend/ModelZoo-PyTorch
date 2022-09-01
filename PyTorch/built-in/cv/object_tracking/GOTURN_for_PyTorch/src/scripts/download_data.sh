#!/bin/bash

if [ -z "$1"  ]
  then
	  echo "No output directory supplied"
	  echo "----------------------------"
	  echo "Example:"
	  echo "./download_data.sh ./output"
	  echo "----------------------------"
	  exit 1
fi
mkdir -p $1

echo "Downloading ImagetNet..."
wget -c https://image-net.org/data/ILSVRC/2014/ILSVRC2014_DET_train.tar $1
wget -c https://image-net.org/data/ILSVRC/2014/ILSVRC2014_DET_bbox_train.tgz $1

echo 'Downloading ALOV dataset...'
wget -c http://isis-data.science.uva.nl/alov/alov300++_frames.zip $1
wget -c http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip $1
echo 'Done'
