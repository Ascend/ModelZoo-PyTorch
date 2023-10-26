# Copyright(C) 2023. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description: Test script for running YoloX model with Ascend Inference Engine
# Author: chenchuwei c00818886
# Create: 2023/10/20

soc_version="Ascend310P3"
data_root="/data/dataset/coco"
batch=1

if [ ! -f "yolox_x.pth" ]; then
    echo "[ERROR] yolox_x.pth not found in current dir, please make sure it exists."
    exit 1
fi

if [ ! -d "YOLOX" ]; then
    echo "[INFO] Preparing YoloX's dependencies"
    git clone https://github.com/Megvii-BaseDetection/YOLOX
    cd YOLOX
    git reset 6880e3999eb5cf83037e1818ee63d589384587bd --hard
    patch -p1 < ../yolox_coco_evaluator.patch
    pip install -v -e .
    cd ..
fi

echo "[INFO] Installing Python dependencies"
apt-get install libprotobuf-dev protobuf-compiler -y
apt-get install libgl1-mesa-glx -y
pip install -r requirements.txt

if [ ! -f "yolox.torchscript.pt" ]; then
    echo "[INFO] Exporting torchscript module"
    cd YOLOX
    python tools/export_torchscript.py --output-name ../yolox.torchscript.pt -n yolox-x -c ../yolox_x.pth
    cd ..
fi

if [ ! -f "yoloxb${batch}_torch_aie.pt" ]; then
    echo "[INFO] AIE Compiling"
    python yolox_export_torch_aie_ts.py \
        --torch-script-path ./yolox.torchscript.pt \
        --batch-size ${batch} \
        --soc-version ${soc_version}
fi

echo "[INFO] Start AIE evaluation"
python yolox_eval.py \
   --dataroot ${data_root} \
   --batch ${batch} \
   --ts ./yoloxb${batch}_torch_aie.pt