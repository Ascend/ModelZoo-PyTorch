# Copyright 2022 Huawei Technologies Co., Ltd
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

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import cv2
import numpy as np
import os
np.set_printoptions(threshold=np.inf)

from mmocr.models.builder import POSTPROCESSOR
from base_postprocessor import BasePostprocessor
from utils import fill_hole, fourier2poly, poly_nms

from mmocr.utils import check_argument

#@POSTPROCESSOR.register_module()
class FCEPostprocessor(BasePostprocessor):
    """Decoding predictions of FCENet to instances.

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        num_reconstr_points (int): The points number of the polygon
            reconstructed from predicted Fourier coefficients.
        text_repr_type (str): Boundary encoding type 'poly' or 'quad'.
        scale (int): The down-sample scale of the prediction.
        alpha (float): The parameter to calculate final scores. Score_{final}
                = (Score_{text region} ^ alpha)
                * (Score_{text center region}^ beta)
        beta (float): The parameter to calculate final score.
        score_thr (float): The threshold used to filter out the final
            candidates.
        nms_thr (float): The threshold of nms.
    """

    def __init__(self,
                 fourier_degree,
                 num_reconstr_points,
                 text_repr_type='poly',
                 alpha=1.0,
                 beta=2.0,
                 score_thr=0.3,
                 nms_thr=0.1,
                 **kwargs):
        super().__init__(text_repr_type)
        self.fourier_degree = fourier_degree
        self.num_reconstr_points = num_reconstr_points
        self.alpha = alpha
        self.beta = beta
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def __call__(self, preds, scale):
        """
        Args:
            preds (list[Tensor]): Classification prediction and regression
                prediction.
            scale (float): Scale of current layer.

        Returns:
            list[list[float]]: The instance boundary and confidence.
        """
        assert isinstance(preds, list)
        assert len(preds) == 2        
        
        cls_pred = preds[0][0]
        tr_pred = cls_pred[0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = cls_pred[2:].softmax(dim=0).data.cpu().numpy()

        reg_pred = preds[1][0].permute(1, 2, 0).data.cpu().numpy()
        x_pred = reg_pred[:, :, :2 * self.fourier_degree + 1]
        y_pred = reg_pred[:, :, 2 * self.fourier_degree + 1:]
        
        score_pred = (tr_pred[1]**self.alpha) * (tcl_pred[1]**self.beta)
        tr_pred_mask = (score_pred) > self.score_thr
        tr_mask = fill_hole(tr_pred_mask)
        tr_contours, _ = cv2.findContours(
            tr_mask.astype(np.uint8), cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE)  # opencv4

        mask = np.zeros_like(tr_mask)
        boundaries = []
        for cont in tr_contours:
            deal_map = mask.copy().astype(np.int8)
            cv2.drawContours(deal_map, [cont], -1, 1, -1)

            score_map = score_pred * deal_map
            score_mask = score_map > 0
            xy_text = np.argwhere(score_mask)
            dxy = xy_text[:, 1] + xy_text[:, 0] * 1j

            x, y = x_pred[score_mask], y_pred[score_mask]
            c = x + y * 1j
            c[:, self.fourier_degree] = c[:, self.fourier_degree] + dxy
            c *= scale

            polygons = fourier2poly(c, self.num_reconstr_points)
            score = score_map[score_mask].reshape(-1, 1)
            polygons = poly_nms(
                np.hstack((polygons, score)).tolist(), self.nms_thr)

            boundaries = boundaries + polygons

        boundaries = poly_nms(boundaries, self.nms_thr)
        
        
        if self.text_repr_type == 'quad':
            new_boundaries = []
            for boundary in boundaries:
                poly = np.array(boundary[:-1]).reshape(-1,
                                                       2).astype(np.float32)
                score = boundary[-1]
                points = cv2.boxPoints(cv2.minAreaRect(poly))
                points = np.int0(points)
                new_boundaries.append(points.reshape(-1).tolist() + [score])
        return boundaries
        
def resize_boundary(boundaries, scale_factor):
    """Rescale boundaries via scale_factor.

    Args:
        boundaries (list[list[float]]): The boundary list. Each boundary
                has :math:`2k+1` elements with :math:`k>=4`.
            scale_factor (ndarray): The scale factor of size :math:`(4,)`.

        Returns:
            list[list[float]]: The scaled boundaries.
        """
    assert check_argument.is_2dlist(boundaries)
    assert isinstance(scale_factor, np.ndarray)
    assert scale_factor.shape[0] == 4

    for b in boundaries:
        sz = len(b)
        check_argument.valid_boundary(b, True)
        b[:sz -1] = (np.array(b[:sz - 1]) *
                (np.tile(scale_factor[:2], int(
                    (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
    return boundaries

    
def get_boundary(scales, score_maps, scale_factor, rescale):
    assert len(score_maps) == len(scales)
    boundaries = []
    for idx, score_map in enumerate(score_maps):
        scale = scales[idx]
        boundaries = boundaries + get_boundary_single(score_map, scale)

        # nms
    boundaries = poly_nms(boundaries, 0.1)

    if rescale:
        boundaries = resize_boundary(boundaries, 1.0 / scale_factor)

    results = dict(boundary_result=boundaries)
    return results

def get_boundary_single(score_map, scale):
    assert len(score_map) == 2
    postprocessor = FCEPostprocessor(fourier_degree = 5,
                     text_repr_type='poly',
                     num_reconstr_points=50,
                     alpha=1.0,
                     beta=2.0,
                     score_thr=0.3)
    return postprocessor(score_map, scale)
    

if __name__ == '__main__':
    #prediction_file_path = '/home/zhangyifan/result/dumpOutput_device0/'
    prediction_file_path = '/home/zhangyifan/result/output2/2022427_7_18_12_711969/'

    container = []
    count = 0
    
    for i in range(501):
         container.append([])
         for j in range(3):
             container[i].append([])
             for k in range(2):
                 container[i][j].append([])
                 
    file_name = '/home/zhangyifan/img_info.txt'
    img_idx = []
    i = 0
    with open(os.path.join(file_name), 'r') as ff:
        temp = ff.readline()
        temp =temp.strip('[')
        temp =temp.strip(']')
        temp =temp.split(',')
        for enum in temp:
            index1 = enum.rfind('_') #rfind()
            index2 = enum.rfind('.') #rfind()
            img_num = enum[index1+1:index2]
            img_idx.append(int(img_num))
            i+=1
    
    for tfile_name in os.listdir(prediction_file_path):
        tmp = tfile_name.split('.')[0]
        index = tmp.rfind('_')
        img_name = tmp[:index]
        index1 = img_name.rfind('_')
        img_name = tmp[:index1]

        index2 = img_name.rfind('_')+1
        flag = int(img_name[index2:])
        
        lines = ''
        with open(os.path.join(prediction_file_path,tfile_name), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                lines = lines+' '+line
            temp = lines.strip().split(" ")
            l = len(temp)
            print(tmp,":",l)
            temp = np.array(temp)
            temp = list(map(float,temp))
            temp = torch.Tensor(temp)
            cont0 = []
            if l == 11360:
                temp = temp.reshape(1,4,40,71)
                container[flag][2][0] = temp
                count += 1
            elif l == 45440:
                temp = temp.reshape(1,4,80,142)
                container[flag][1][0] = temp
                count += 1
            elif l == 181760:
                temp = temp.reshape(1,4,160,284)
                container[flag][0][0] = temp
                count += 1
            elif l == 62480:
                temp = temp.reshape(1,22,40,71)
                container[flag][2][1] = temp
                count += 1
            elif l == 249920:
                temp = temp.reshape(1,22,80,142)
                container[flag][1][1] = temp
                count += 1
            elif l == 999680:
                temp = temp.reshape(1,22,160,284)
                container[flag][0][1] = temp
                count += 1         
                
    postprocess = FCEPostprocessor(fourier_degree = 5,
                     text_repr_type='poly',
                     num_reconstr_points=50,
                     alpha=1.0,
                     beta=2.0,
                     score_thr=0.3)
    scale = 5
    scales = (8, 16, 32)
    rescale = True
    scale_factor = np.array([1.765625 , 1.7652777, 1.765625 , 1.7652777])
    for i in range(0,500):
        idx = img_idx[i]
        score_maps = container[idx]    
        result = get_boundary(scales, score_maps, scale_factor, rescale)
        f=open("/home/zhangyifan/boundary_results_4.txt","a+")
        f.writelines(str(result)+'\n')
