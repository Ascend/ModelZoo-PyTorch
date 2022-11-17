#!/bin/bash
# VOT
git clone https://github.com/jvlmdr/trackdat.git
cd trackdat
VOT_YEAR=2018 bash scripts/download_vot.sh dl/vot2018
bash scripts/unpack_vot.sh dl/vot2018 ../VOT2018
cp dl/vot2018/list.txt ../VOT2018/
cd .. && rm -rf ./trackdat

# json file for eval toolkit
wget http://www.robots.ox.ac.uk/~qwang/VOT2018.json
