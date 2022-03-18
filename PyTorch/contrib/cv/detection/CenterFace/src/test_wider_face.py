# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import sys
import os
import scipy.io as sio
import cv2


path = os.path.dirname(__file__)
CENTERNET_PATH = os.path.join(path,'../src/lib')
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts_pose import opts
from datasets.dataset_factory import get_dataset


def test_img(MODEL_PATH):
    debug = 1            # draw and show the result image
    TASK = 'multi_pose'  
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h).split(' '))

    detector = detector_factory[opt.task](opt)

    img = '../readme/000388.jpg'
    ret = detector.run(img)['results']

def test_vedio(MODEL_PATH, vedio_path=None):
    debug = -1            # return the result image with draw
    TASK = 'multi_pose'  
    vis_thresh = 0.45
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --input_h {} --input_w {} --vis_thresh {}'.format(
        TASK, MODEL_PATH, debug, intput_w, input_h, vis_thresh).split(' '))
    detector = detector_factory[opt.task](opt)

    vedio = vedio_path if vedio_path else 0
    cap = cv2.VideoCapture(vedio)
    while cap.isOpened():
        det = cap.grab()
        if det:
            flag, frame = cap.retrieve()
            res = detector.run(frame)
            cv2.imshow('face detect', res['plot_img'])

        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def test_wider_Face(MODEL_PATH, project_path):
    Path = project_path + '/data/wider_face/WIDER_val/images/'
    wider_face_mat = sio.loadmat(project_path + '/data/wider_face/ground_truth/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_path = project_path + '/output/widerface/'

    debug = 0            # return the detect result without show
    threshold = 0.05
    TASK = 'multi_pose'  
    input_h, intput_w = 800, 800
    opt = opts().init('--task {} --load_model {} --debug {} --vis_thresh {} --input_h {} --input_w {}'.format(
        TASK, MODEL_PATH, debug, threshold, input_h, intput_w).split(' '))
    detector = detector_factory[opt.task](opt)
    
    for index, event in enumerate(event_list):
        file_list_item = file_list[index][0]
        im_dir = event[0][0]
        if not os.path.exists(save_path + im_dir):
            os.makedirs(save_path + im_dir)
        for num, file in enumerate(file_list_item):
            im_name = file[0][0]
            zip_name = '%s/%s.jpg' % (im_dir, im_name)
            print(os.path.join(Path, zip_name))
            img_path = os.path.join(Path, zip_name)
            
            dets = detector.run(img_path)['results']
            
            f = open(save_path + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir, im_name)))
            f.write('{:d}\n'.format(len(dets)))
            for b in dets[1]:
                x1, y1, x2, y2, s = b[0], b[1], b[2], b[3], b[4]
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1, y1, (x2 - x1 + 1), (y2 - y1 + 1), s))
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))


if __name__ == '__main__':
    project_path = os.path.abspath(os.path.dirname(os.getcwd()))
    MODEL_PATH = project_path + "/exp/multi_pose/mobilev2_10/model_best.pth"
    #MODEL_PATH1 = '/root/wc/CenterFace-NPU/exp/multi_pose/mobilev2_10/model_5.pth'
    #test_img(MODEL_PATH)
    #test_vedio(MODEL_PATH)
    test_wider_Face(MODEL_PATH, project_path)