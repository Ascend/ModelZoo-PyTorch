# Copyright 2022 Huawei Technologies Co., Ltd
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
    
import cv2
import numpy as np
import h5py


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def affine_transform(pt, trans_mat):
    """Apply an affine transformation to the points.

    Args:
        pt (np.ndarray): a 2 dimensional point to be transformed
        trans_mat (np.ndarray): 2x3 matrix of an affine transform

    Returns:
        np.ndarray: Transformed points. 
    """
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(trans_mat, new_pt)

    return new_pt[:2]
    
    
def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point. 
    """
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False,
                         scale_ratio=200.0):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
        scale_ratio(float): pixel std of MPII is 200

    Returns:
        np.ndarray: The transform matrix.
    """ 
    scale_tmp = scale * scale_ratio

    shift = np.array(shift)
    src_w = scale_tmp
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans    
    

def get_img(args):
    """
    load image set
    """
    h5file = h5py.File(args.annot_dir, "r") # read lable file
    img_nums = 2958 # there are 2958 images in valid.h5
    for i in range(img_nums):
        imgName = h5file["imgname"][i].decode("UTF-8")
        imgPath = "%s/%s" % (args.img_dir, imgName)
        imgBGR = cv2.imread(imgPath, cv2.IMREAD_COLOR) 
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB) # transform image format from BGR to RGB
        
        image_size = (384, 384) # input shape of Hourglass model
        
        c = h5file["center"][i]
        s = h5file["scale"][i]
        r = 0 # rotation
        
        trans = get_affine_transform(c, s, r, image_size) # get affine transform matrix
        imgTrans = cv2.warpAffine(
                    imgRGB,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags = cv2.INTER_LINEAR)
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        imgMeanStd = (imgTrans.astype(np.float32) / 255 - mean) / std

        imgEval = imgMeanStd.reshape((1,) + imgMeanStd.shape)
        imgEval = imgEval.transpose(0, 3, 1, 2) # turn image format into NCHW(1*3*384*384)

        kp = h5file["part"][i]
        vis = h5file["visible"][i] # if visible vis equals to 1, else vis equals to 0
        kp2 = np.insert(kp, 2, vis, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2

        n = h5file["normalize"][i]

        yield kps, imgEval, c, s, n, imgName

    