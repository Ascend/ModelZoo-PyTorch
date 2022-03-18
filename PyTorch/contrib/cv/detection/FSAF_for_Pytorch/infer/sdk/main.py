# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import json
import os

import cv2
import numpy as np
from skimage import io

from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi


def parse_arg():
    """
    Parse argument from the command line.
    """
    parser = argparse.ArgumentParser(description="FSAF infer")
    parser.add_argument("-d", "--dataset", default="../data/input/val2017")
    parser.add_argument("-p", "--pipeline", default= "../data/config/fsaf.pipeline")
    parser.add_argument("--test_annotation", default="../data/input/coco2017.info")

    parser.add_argument("--det_results_path", default="./output/infer_result")
    parser.add_argument("--show_results_path", default="./output/show_result")
    parser.add_argument("--net_out_num", default=2)
    parser.add_argument("--net_input_width", default=1216)
    parser.add_argument("--net_input_height", default=800)
    parser.add_argument("--prob_thres", default=0.05)
    parser.add_argument("--bbox_thres", default=0.5)
    parser.add_argument("--ifShowDetObj", action="store_true", help="if input the para means True, neither False.")
    
    flags = parser.parse_args()
    return flags


def get_dataset(path):
    """
    Get all images under this path.
    :param path: string, the path of dataset
    :return: a iteratable object
    """
    for root, dirs, files in os.walk(path):
        for file_name in files: 
            if file_name.endswith('jpg') or file_name.endswith('JPG'):
                yield os.path.join(path, file_name)
        break


def get_stream_manager(pipeline_path):
    """
    Init and create stream manager by a pipeline. 
    :param pipeline_path: string, the pipeline config file path
    :return: StreamManagerApi object
    """
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    with open(pipeline_path, 'rb') as f:
        pipeline_content = f.read()

    ret = stream_manager_api.CreateMultipleStreams(pipeline_content)
    if ret != 0:
        print("Failed to create stream, ret=%s" % str(ret))
        exit()
    return stream_manager_api


def do_infer_image(stream_manager_api, image_path):
    """
    Infer image by using stream manager api.
    :param stream_manager_api: StreamManagerApi object
    :param image_path: string, the image path
    :return: ndarray bbox (1,100,5); ndarray, label (1,100)
    """
    stream_name = b'im_fsaf'
    data_input = MxDataInput()
    with open(image_path, 'rb') as f:
        data_input.data = f.read()

    # send image data to pipeline
    unique_id = stream_manager_api.SendData(stream_name, 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    # get infer result
    infer_result = stream_manager_api.GetResult(stream_name, unique_id)
    if infer_result.errorCode != 0:
        print(f"GetResult error.errorCode={infer_result.errorCode}", 
              f"errorMsg={infer_result.data.decode()}")
        exit()

    infer_result_json = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_json['metaData'][0]['content'])
    # print the infer data
    print(infer_result.data.decode())
    infer_result_json = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_json['metaData'][0]['content'])

    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    bboxes = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    bboxes = np.reshape(bboxes, tensor_shape[1:])
    
    print("---------------------------bboxes---------------------------")
    print(bboxes, '\n')

    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][1]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    labels = np.frombuffer(base64.b64decode(data_str), dtype=np.int64)
    labels = np.reshape(labels, tensor_shape[1:])

    print("---------------------------labels---------------------------")
    print(labels, '\n')

    # [bboxes,labels]  (1,100,5);(1,100)
    return bboxes, labels

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def postprocess_bboxes(bboxes, image_size, net_input_width, net_input_height):
    """
    Post process bbox, Scale the bbox to the origin image size. 
    :param bboxes: ndarray, 100*5
    :param image_size: list, have 2 elements, [origin width, origin height]
    :param net_input_width: int, 1216
    :param net_input_height: int, 800
    :return: bbox after post processing
    """
    w = image_size[0]
    h = image_size[1]
    scale = min(net_input_width / w, net_input_height / h)

    pbboxes = bboxes.copy()
    # cal predict box on the image src
    pbboxes[:, 0] = pbboxes[:, 0] / scale
    pbboxes[:, 1] = pbboxes[:, 1] / scale
    pbboxes[:, 2] = pbboxes[:, 2] / scale
    pbboxes[:, 3] = pbboxes[:, 3] / scale

    # make pbboxes value in valid range
    pbboxes[:, 0] = np.maximum(pbboxes[:, 0], 0)
    pbboxes[:, 1] = np.maximum(pbboxes[:, 1], 0)
    pbboxes[:, 2] = np.minimum(pbboxes[:, 2], net_input_width)
    pbboxes[:, 3] = np.minimum(pbboxes[:, 3], net_input_height)

    return pbboxes


def main(args):
    # for debug count
    i = 0
    path = args.dataset
    det_results_path = args.det_results_path
    show_results_path = args.show_results_path

    # recursively create result directories 
    os.makedirs(det_results_path, exist_ok=True)
    os.makedirs(show_results_path, exist_ok=True)
    
    # create stream manager
    stream_manager_api = get_stream_manager(args.pipeline)

    # key: string, image name
    # value: 3-tuple, (width, height, image path)
    img_info_dict = dict()
    with open(args.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_info_dict[img_name] = (img_width, img_height, img_file_path)

    for img_path in get_dataset(path):
        # add i
        i += 1
        # read image
        curImage = io.imread(img_path)
        # ensure H*W*C, C=3
        if len(curImage.shape) == 3:
            if curImage.shape[2] != 3:
                continue
        file_name1 = os.path.basename(img_path)
        # drop suffix
        file_name = file_name1.split('.')[0]
        print(file_name1)

        # [bboxes,labels]  (1,100,5);(1,100)
        # bboxes (x0, y0, x1, y1,confidence）
        # labels (class)
        bboxes, labels = do_infer_image(stream_manager_api, img_path)

        bboxes = np.reshape(bboxes, [100, 5])
        labels = np.reshape(labels, [100, 1])

        current_img_info = img_info_dict[file_name]
        current_img_size = current_img_info[:2]
        bboxes = postprocess_bboxes(bboxes, current_img_size, args.net_input_width, args.net_input_height)

        res_buff = []
        res_buff.append(bboxes)
        res_buff.append(labels)
        # predbox [100, 6]
        predbox = np.concatenate(res_buff, axis=1)
        
        # current_img_info (640, 427, '/home/wyh/data/coco/val2017/000000274066.jpg')
        print("[TEST]---------------------------imgInfo{}".format(current_img_info))

        # whether show the visualizition
        if args.ifShowDetObj == True:
            imgCur = cv2.imread(img_path)
           
        det_results_str = ''

        # iterator infer result of every image
        for idx, class_ind in enumerate(predbox[:,5]):
            if float(predbox[idx][4]) < float(args.prob_thres):
                continue
            # skip negative class index
            if class_ind < 0 or class_ind > 80:
                continue

            class_name = CLASSES[int(class_ind)]
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, str(predbox[idx][4]), predbox[idx][0],
                                                            predbox[idx][1], predbox[idx][2], predbox[idx][3])
            # draw bbox on the origin image
            if args.ifShowDetObj == True:
                imgCur=cv2.rectangle(imgCur, (int(predbox[idx][0]), int(predbox[idx][1])), 
                                    (int(predbox[idx][2]), int(predbox[idx][3])), (0,255,0), 1)
                imgCur = cv2.putText(imgCur, class_name + '|' + str(predbox[idx][4]), 
                                    (int(predbox[idx][0]), int(predbox[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (0, 0, 255), 1)

        # save visualizition result   
        if args.ifShowDetObj == True:
            print(os.path.join(show_results_path, file_name +'.jpg'))
            cv2.imwrite(os.path.join(show_results_path, file_name +'.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY),70])

        # save infer result
        det_results_file = os.path.join(det_results_path, file_name + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)

        print(det_results_str)
        print(i)
     
    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    args = parse_arg()
    main(args)