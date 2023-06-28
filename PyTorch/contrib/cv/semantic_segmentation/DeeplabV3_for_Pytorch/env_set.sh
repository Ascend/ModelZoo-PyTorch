cur_path=`pwd`

echo y | pip3 uninstall mmsegmentation
echo y | pip3 uninstall mmcv-full

chmod +x ${cur_path}/tools/*sh
chmod +x ${cur_path}/test/*sh
source ${cur_path}/test/env_npu.sh

pip3 install -r requirements.txt
pip3 install -e .
cd ..
git clone -b v1.3.9 --depth=1 https://github.com/open-mmlab/mmcv.git
export MMCV_WITH_OPS=1
export MAX_JOBS=8
cd mmcv
python3 setup.py build_ext
python3 setup.py develop
pip3.7 uninstall opencv-python
pip3.7 install opencv-python-headless

cd ${cur_path}
/bin/cp -f mmcv_need/_functions.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/scatter_gather.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/dist_utils.py ../mmcv/mmcv/runner/
/bin/cp -f mmcv_need/distributed.py ../mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/epoch_based_runner.py ../mmcv/mmcv/runner/
