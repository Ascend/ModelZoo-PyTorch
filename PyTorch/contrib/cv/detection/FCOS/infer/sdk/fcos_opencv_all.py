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
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi
from skimage import io


def parse_arg():
    parser = argparse.ArgumentParser(description="FCOS infer")
    parser.add_argument("--dataset", default="../data/input/COCO2017/val2017")
    parser.add_argument("--pipeline", default="../data/config/fcos.pipeline")
    parser.add_argument("--test_annotation",
                        default="../data/input/COCO2017/coco2017.info")
    parser.add_argument("--det_results_path", default="./data/infer_result")
    parser.add_argument("--show_results_path", default="./data/show_result")
    parser.add_argument("--net_input_width", default=1333)
    parser.add_argument("--net_input_height", default=800)
    parser.add_argument("--prob_thres", default=0.05)
    parser.add_argument(
        "--ifShowDetObj",
        default="True",
        action="store_true",
        help="if input the para means True, neither False.")

    flags = parser.parse_args()
    return flags


def get_dataset(path):
    """
    This function is getting data from dataset on the path.

    :param path: a string of dataset path

    """
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith('jpg') or file_name.endswith('JPG'):
                yield os.path.join(path, file_name)
        break


def get_stream_manager(pipeline_path):
    """
    This function is using stream_manager_api.

    :param pipeline_path: a string of pipeline path
    :return: a stream manager

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
    This function is executing the inference of images.

    :param stream_manager_api: a stream manager
    :param image_path: a string of image path
    :return: bbox, labels

    bbox,labels: (1,100,5),(1,100)
    The model has two output tensors:
        bbox:(x0, y0, x1, y1,confidence) 
        #the upper left and lower right coordinates of the detection boxes
        labels: probability of 80 classes
    """
    stream_name = b'im_fcos'
    data_input = MxDataInput()
    with open(image_path, 'rb') as f:
        data_input.data = f.read()

    unique_id = stream_manager_api.SendData(stream_name, 0, data_input)
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    infer_result = stream_manager_api.GetResult(stream_name, unique_id)
    if infer_result.errorCode != 0:
        print(f"GetResult error. errorCode={infer_result.errorCode},"
              f"errorMsg={infer_result.data.decode()}")
        exit()

    infer_result_json = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_json['metaData'][0]['content'])
    # print the infer result
    print(infer_result.data.decode())
    infer_result_json = json.loads(infer_result.data.decode())
    content = json.loads(infer_result_json['metaData'][0]['content'])
    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][0]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    bbox = np.frombuffer(base64.b64decode(data_str), dtype=np.float32)
    bbox = np.reshape(bbox, tensor_shape[1:])
    # [bbox,labels]  (1,100,5);(1,100)

    print("---------------------------bbox---------------------------")
    print(bbox)
    print()
    print(bbox.shape)
    print("-----------------------------------------------------------------")

    tensor_vec = content['tensorPackageVec'][0]['tensorVec'][1]
    data_str = tensor_vec['dataStr']
    tensor_shape = tensor_vec['tensorShape']
    labels = np.frombuffer(base64.b64decode(data_str), dtype=np.int64)
    labels = np.reshape(labels, tensor_shape[1:])
    print("---------------------------labels---------------------------")
    print(labels)
    print()
    print(labels.shape)
    print("-----------------------------------------------------------------")
    return bbox, labels


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


def coco_postprocess(bbox: np.ndarray, image_size,
                     net_input_width, net_input_height):
    """
    This function is postprocessing for FCOS output.

    Before calling this function, reshape the raw output of FCOS to
    following form
        numpy.ndarray:
            [x0, y0, x1, y1, confidence, probability of 80 classes]
        shape: (100,)
    The postprocessing restore the bounding rectangles of FCOS output
    to origin scale and filter with non-maximum suppression.

    :param bbox: a numpy array of the FCOS output
    :param image_path: a string of image path
    :return: three list for best bound, class and score
    """
    w = image_size[0]
    h = image_size[1]
    scale = min(net_input_width / w, net_input_height / h)

    pad_w = net_input_width - w * scale
    pad_h = net_input_height - h * scale
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    # cal predict box on the image src
    pbox = bbox

    pbox[:, 0] = (bbox[:, 0] - pad_left) / scale
    pbox[:, 1] = (bbox[:, 1] - pad_top) / scale
    pbox[:, 2] = (bbox[:, 2] - pad_left) / scale
    pbox[:, 3] = (bbox[:, 3] - pad_top) / scale

    # make pbboxes value in valid range
    pbox[:, 0] = np.maximum(pbox[:, 0], 0)
    pbox[:, 1] = np.maximum(pbox[:, 1], 0)
    pbox[:, 2] = np.minimum(pbox[:, 2], w)
    pbox[:, 3] = np.minimum(pbox[:, 3], h)

    return pbox


def main(args):
    i = 0
    path = args.dataset
    print(args.ifShowDetObj, type(args.ifShowDetObj))
    det_results_path = args.det_results_path
    show_results_path = args.show_results_path
    os.makedirs(det_results_path, exist_ok=True)
    os.makedirs(show_results_path, exist_ok=True)
    stream_manager_api = get_stream_manager(args.pipeline)
    img_size_dict = dict()
    with open(args.test_annotation)as f:
        for line in f.readlines():
            temp = line.split(" ")
            img_file_path = temp[1]
            img_name = temp[1].split("/")[-1].split(".")[0]
            img_width = int(temp[2])
            img_height = int(temp[3])
            img_size_dict[img_name] = (img_width, img_height, img_file_path)

    for img_path in get_dataset(path):
        image_1 = io.imread(img_path)
        if len(image_1.shape) == 3:
            if image_1.shape[2] != 3:
                continue
        file_name1 = os.path.basename(img_path)
        file_name = file_name1.split('.')[0]
        print(file_name1)
        delete_img_name = ['000000374551.jpg', '000000003661.jpg',
                           '000000309391.jpg', '000000070254.jpg']
        if file_name1 in delete_img_name:
            continue

        bbox, labels = do_infer_image(stream_manager_api, img_path)

        res_buff = []
        res_buff.append(bbox)
        labels = np.reshape(labels, [100, 1])
        res_buff.append(labels)
        res_tensor = np.concatenate(res_buff, axis=1)
        current_img_size = img_size_dict[file_name]
        print("[TEST]---------------------------concat{} imgsize{}".format(
            len(res_tensor), current_img_size))
        predbox = coco_postprocess(
            res_tensor, current_img_size, args.net_input_width, args.net_input_height)

        if args.ifShowDetObj == True:
            imgCur = cv2.imread(img_path)

        det_results_str = ''
        for idx, class_ind in enumerate(predbox[:, 5]):
            if float(predbox[idx][4]) < float(args.prob_thres):
                continue
            # skip negative class index
            if class_ind < 0 or class_ind > 80:
                continue

            class_name = CLASSES[int(class_ind)]
            det_results_str += "{} {} {} {} {} {}\n".format(class_name, str(predbox[idx][4]), predbox[idx][0],
                                                            predbox[idx][1], predbox[idx][2], predbox[idx][3])
            if args.ifShowDetObj == True:
                imgCur = cv2.rectangle(imgCur, (int(predbox[idx][0]), int(predbox[idx][1])),
                                       (int(predbox[idx][2]), int(predbox[idx][3])), (0, 255, 0), 1)
                imgCur = cv2.putText(imgCur, class_name+'|'+str(predbox[idx][4]),
                                     (int(predbox[idx][0]), int(predbox[idx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Image, text content, coordinates, font, size, color and font thickness.

        if args.ifShowDetObj == True:
            print(os.path.join(show_results_path, file_name + '.jpg'))
            cv2.imwrite(os.path.join(show_results_path, file_name +
                        '.jpg'), imgCur, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        det_results_file = os.path.join(det_results_path, file_name + ".txt")
        with open(det_results_file, "w") as detf:
            detf.write(det_results_str)
        print(det_results_str)
        i = i+1
        print(i)

    stream_manager_api.DestroyAllStreams()


if __name__ == "__main__":
    args = parse_arg()
    main(args)
