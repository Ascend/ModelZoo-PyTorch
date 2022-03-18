echo 'start to prepare aipp test input'
python3 gen_dataset_info.py \
--src_path_testA=./datasets/maps/testA/          \
--save_pathTestA_dst=datasetsDst/maps/testA/    \
--dataTestA_infoName=testA_prep.info          \
--src_path_testB=./datasets/maps/testB/         \
--save_pathTestB_dst=./datasetsDst/maps/testB/  \
--dataTestA_infoName=testB_prep.info
