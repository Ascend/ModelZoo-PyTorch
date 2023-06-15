curPath=$(dirname $(readlink -f "$0"))

echo y | pip3 uninstall mmdet
echo y | pip3 uninstall mmpycocotools
echo y | pip3 uninstall pycocotools

chmod +x ${curPath}/tools/*sh
mkdir -p ${curPath}/data
mkdir logs

code_url=`sed '/^code_url=/!d;s/.*=//' url.ini`
git clone -b v1.2.7 --depth=1 ${code_url}
export MMCV_WITH_OPS=1
export MAX_JOBS=8

source ./test/env_npu.sh
cd mmcv
python3 setup.py build_ext
python3 setup.py develop

cd ${curPath}
pip3 install -r requirements/build.txt
pip3 install -v -e .
/bin/cp -f mmcv_need/_functions.py mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/builder.py mmcv/mmcv/runner/optimizer/
/bin/cp -f mmcv_need/data_parallel.py mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/dist_utils.py mmcv/mmcv/runner/
/bin/cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/
/bin/cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/
