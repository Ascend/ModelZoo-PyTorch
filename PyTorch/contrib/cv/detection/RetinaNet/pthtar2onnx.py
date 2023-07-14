# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
import detectron2
import onnx
import torch
from PIL import Image
import cv2
from urllib.request import urlretrieve

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export import export_onnx_model, export_caffe2_model
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from detectron2.engine.defaults import DefaultPredictor

def main():   
    cfg = get_cfg()
    # add project-specific config 
    cfg.merge_from_file('configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml')
    cfg.freeze()

    current_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(current_path, './url.ini'), 'r') as _f:
        _content = _f.read()
        IMAGE_URL = _content.split('image_url=')[1].split('\n')[0]
    urlretrieve(IMAGE_URL, 'tmp.jpg')
    im = cv2.imread("tmp.jpg")
    model = build_model(cfg)
    model.eval()
    
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load("/root/txyWorkSpace/detectron2/output1/model_final.pth")
    aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                           cfg.INPUT.MAX_SIZE_TEST)
    height, width = im.shape[:2]
    image = aug.get_transform(im).apply_image(im)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image, "height": height, "width": width}

    # Export to Onnx model
    onnxModel = export_onnx_model(cfg, model, [inputs])
    onnx.save(onnxModel, "RetinaNet_npu.onnx")
    
if __name__ == "__main__":
    main()
