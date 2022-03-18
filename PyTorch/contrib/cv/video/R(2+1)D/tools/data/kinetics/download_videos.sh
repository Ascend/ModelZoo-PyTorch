#!/usr/bin/env bash

# set up environment
# conda env create -f environment.yml
# source activate kinetics
# pip install --upgrade youtube-dl

activate kinetics


# DATASET=kinetics400
# if [ "$DATASET" == "kinetics400" ] || [ "$1" == "kinetics600" ] || [ "$1" == "kinetics700" ]; then
#         echo "We are processing $DATASET"
# else
#         echo "Bad Argument, we only support kinetics400, kinetics600 or kinetics700"
#         exit 0
# fi

DATA_DIR="../../../data/kinetics400"
ANNO_DIR="../../../data/kinetics400/annotations"
python download.py ${ANNO_DIR}/kinetics_train.csv ${DATA_DIR}/videos_train
python download.py ${ANNO_DIR}/kinetics_val.csv ${DATA_DIR}/videos_val

deactivate kinetics
# conda remove -n kinetics --all
