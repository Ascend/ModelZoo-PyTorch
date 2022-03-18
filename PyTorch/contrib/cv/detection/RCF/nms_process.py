import os
import cv2
import numpy as np
from scipy.io import loadmat

from impl.toolbox import conv_tri, grad2
from ctypes import *


# NOTE:
#    In NMS, `if edge < interp: out = 0`, I found that sometimes edge is very close to interp.
#    `edge = 10e-8` and `interp = 11e-8` in C, while `edge = 10e-8` and `interp = 9e-8` in python.
#    ** Such slight differences (11e-8 - 9e-8 = 2e-8) in precision **
#    ** would lead to very different results (`out = 0` in C and `out = edge` in python). **
#    Sadly, C implementation is not expected but needed :(
solver = cdll.LoadLibrary("cxx/lib/solve_csa.so")
c_float_pointer = POINTER(c_float)
solver.nms.argtypes = [c_float_pointer, c_float_pointer, c_float_pointer, c_int, c_int, c_float, c_int, c_int]


def nms_process_one_image(image, save_path=None, save=True):
    """"
    :param image: numpy array, edge, model output
    :param save_path: str, save path
    :param save: bool, if True, save .png
    :return: edge
    NOTE: in MATLAB, uint8(x) means round(x).astype(uint8) in numpy
    """

    if save and save_path is not None:
        assert os.path.splitext(save_path)[-1] == ".png"
    edge = conv_tri(image, 1)
    ox, oy = grad2(conv_tri(edge, 4))
    oxx, _ = grad2(ox)
    oxy, oyy = grad2(oy)
    ori = np.mod(np.arctan(oyy * np.sign(-oxy) / (oxx + 1e-5)), np.pi)
    out = np.zeros_like(edge)
    r, s, m, w, h = 1, 5, float(1.01), int(out.shape[1]), int(out.shape[0])
    solver.nms(out.ctypes.data_as(c_float_pointer),
               edge.ctypes.data_as(c_float_pointer),
               ori.ctypes.data_as(c_float_pointer),
               r, s, m, w, h)
    edge = np.round(out * 255).astype(np.uint8)
    if save:
        cv2.imwrite(save_path, edge)
    return edge


def nms_process(model_name_list, result_dir, save_dir, key=None, file_format=".mat"):
    if not isinstance(model_name_list, list):
        model_name_list = [model_name_list]
    assert file_format in {".mat", ".npy"}
    assert os.path.isdir(result_dir)

    for model_name in model_name_list:
        model_save_dir = os.path.join(save_dir, model_name)
        # print(model_save_dir)
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)

        for file in os.listdir(result_dir):
            save_name = os.path.join(model_save_dir, "{}.png".format(os.path.splitext(file)[0]))
            # print(save_name)
            if os.path.isfile(save_name):
                continue

            if os.path.splitext(file)[-1] != file_format:
                continue
            abs_path = os.path.join(result_dir, file)
            print(abs_path)
            if file_format == ".mat":
                assert key is not None
                image = loadmat(abs_path)[key]
            elif file_format == ".npy":
                image = np.load(abs_path)
            else:
                raise NotImplementedError
            nms_process_one_image(image, save_name, True)


if __name__ == '__main__':
    nms_process("hed", "hed_result", "NMS_RESULT_FOLDER", key="result")
