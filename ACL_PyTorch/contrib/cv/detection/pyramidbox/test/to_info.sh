 #!/bin/bash
cd ../
#source npu_set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3.7 get_info.py bin ./data1000_1 ./pyramidbox_pre_bin_1000_1.info 1000 1000
python3.7 get_info.py bin ./data1000_2 ./pyramidbox_pre_bin_1000_2.info 1000 1000
