import argparse
import os

import numpy as np
import torch

from modules.parse_polys import parse_polys
import re
import tqdm
import os
import sys
import struct


# bin文件格式转为numpy
def bin2np(binName,binShape):
    size = os.path.getsize(binName)     # size 是字节大小
    binfile = open(binName, 'rb')   
    Len = int(size / 4)                 # 4个字节=float32 类型
    res=[]          
    for i in range(Len):
        data = binfile.read(4)              #将4个字节取出作为 float
        num = struct.unpack('f', data)
        res.append(num[0])
    
    binfile.close()
    
    dim_res = np.array(res).reshape(binShape)
    
    return dim_res


# bin 文件转回 tensor
def postprocess(bin_folder, output_folder):
    #pbar = tqdm.tqdm(os.listdir(bin_folder), desc='Convert', ncols=80)
    #for image_name in pbar:
    #prefix = image_name[:image_name.rfind('.')]
    
    #提取出 img_1'
    preNum = 1
    while (preNum < 501 ):
    
        scale_x = 2240 / 1280
        scale_y = 1248 / 720
        
        preName = "img_" + str(preNum) 
        confBin = bin_folder + preName + "_1.bin"
        disBin = bin_folder + preName + "_2.bin"
        angleBin = bin_folder+ preName + "_3.bin"
        preNum += 1
        
        print("deal bin file = ",confBin)
        confidence = torch.tensor( bin2np(confBin,(1,1,312,560)))
        distances = torch.tensor( bin2np(disBin,(1,4,312,560)))
        angle = torch.tensor( bin2np(angleBin,(1,1,312,560)))
        
        confidence = torch.sigmoid(confidence).squeeze().data.cpu().numpy()
        distances = distances.squeeze().data.cpu().numpy()
        angle = angle.squeeze().data.cpu().numpy()
        
        
        
        polys = parse_polys(confidence, distances, angle, 0.95, 0.3)  # , img=orig_scaled_image)
        with open('{}'.format(os.path.join(output_folder, 'res_{}.txt'.format(preName))), 'w') as f:
            for id in range(polys.shape[0]):
                f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                    int(polys[id, 0] / scale_x), int(polys[id, 1] / scale_y), int(polys[id, 2] / scale_x),
                    int(polys[id, 3] / scale_y),
                    int(polys[id, 4] / scale_x), int(polys[id, 5] / scale_y), int(polys[id, 6] / scale_x),
                    int(polys[id, 7] / scale_y)
                ))
        
        print("----get output_folder----")
        #pbar.set_postfix_str(image_name, refresh=False)



if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-folder', type=str, required=True, help='path to the folder with test images')
    parser.add_argument('--output-folder', type=str, default='fots_test_results',
                        help='path to the output folder with result labels')
    args = parser.parse_args()
    '''
    output_folder = "./outPost/"
    bin_folder="./result/dumpOutput_device1/"

    postprocess(bin_folder, output_folder)
    
    
    
    