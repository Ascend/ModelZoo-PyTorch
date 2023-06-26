#!/bin/bash
# VOT
jvlmdr_url=`sed '/^jvlmdr_url=/!d;s/.*=//' ../url.ini`
git clone ${jvlmdr_url}
cd trackdat
VOT_YEAR=2018 bash scripts/download_vot.sh dl/vot2018
bash scripts/unpack_vot.sh dl/vot2018 ../VOT2018
cp dl/vot2018/list.txt ../VOT2018/
cd .. && rm -rf ./trackdat

# json file for eval toolkit
vot2018_url=`sed '/^vot2018_url=/!d;s/.*=//' ../url.ini`
wget ${vot2018_url}
