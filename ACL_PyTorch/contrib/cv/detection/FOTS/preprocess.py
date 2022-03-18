import argparse
import os

import cv2
import numpy as np
import torch



import re
import tqdm


def preprocess(images_folder, output_folder):
    pbar = tqdm.tqdm(os.listdir(images_folder), desc='Test', ncols=80)
    for image_name in pbar:
     #  prefix = image_name[:image_name.rfind('.')]
        image = cv2.imread(os.path.join(images_folder, image_name), cv2.IMREAD_COLOR)
        # due to bad net arch sizes have to be mult of 32, so hardcode it
        scale_x = 2240 / image.shape[1]  # 2240 # 1280    1.75
        scale_y = 1248 / image.shape[0]  # 1248 # 720     1.73333
        scaled_image = cv2.resize(image, dsize=(0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
     #  orig_scaled_image = scaled_image.copy()

        scaled_image = scaled_image[:, :, ::-1].astype(np.float32)
        scaled_image = (scaled_image / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image_tensor = torch.from_numpy(np.expand_dims(np.transpose(scaled_image, axes=(2, 0, 1)), axis=0)).float()
        
        img = np.array(image_tensor).astype(np.float32)
        
        img.tofile(os.path.join(output_folder, image_name.split('.')[0] + ".bin"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-folder', type=str, required=True, help='path to the folder with test images')
    parser.add_argument('--output-folder', type=str, default='fots_test_results',
                        help='path to the output folder with result labels')

    args = parser.parse_args()

    preprocess(args.images_folder, args.output_folder)


