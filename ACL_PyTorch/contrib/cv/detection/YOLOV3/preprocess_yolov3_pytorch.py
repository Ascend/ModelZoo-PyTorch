import sys
import os
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
        # dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    # elif scaleFill:  # stretch
        # dw, dh = 0.0, 0.0
        # new_unpad = (new_shape[1], new_shape[0])
        # ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def yolov3_onnx(src_info, output_path):
    in_files = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(src_info, 'r') as file:
        contents = file.read().split('\n')
    for i in contents[:-1]:
        in_files.append(i.split()[1])

    i = 0
    for file in in_files:
        i = i + 1
        print(file, "====", i)
        img0 = cv2.imread(file)
        # Padded resize
        img = letterbox(img0, new_shape=416)[0]
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        image_np = np.array(img, dtype=np.float32)
        image_np /= 255.0
        # image_np = np.transpose(image_np, (2, 0, 1))  # HWC -> CHW
        image_np_expanded = np.expand_dims(image_np, axis=0)  # NCHW
        # Focus
        print("shape:", image_np_expanded.shape)
        img_numpy = np.ascontiguousarray(image_np_expanded)

        # save img_tensor as binary file for om inference input
        temp_name = file[file.rfind('/') + 1:]
        img_numpy.tofile(os.path.join(output_path, temp_name.split('.')[0] + ".bin"))


if __name__ == "__main__":
    src_info = os.path.abspath(sys.argv[1])
    bin_path = os.path.abspath(sys.argv[2])
    yolov3_onnx(src_info, bin_path)


