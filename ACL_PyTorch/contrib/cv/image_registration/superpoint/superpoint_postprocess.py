# Copyright 2022 Huawei Technologies Co., Ltd
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
import ast
import matplotlib
matplotlib.use('Agg') # solve error of tk
import argparse
import time
import yaml
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

## torch
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from numpy.linalg import inv
from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import compute_repeatability
import cv2
from utils.logging import *


#from utils.losses import extract_patches
#from Train_model_heatmap import Train_model_heatmap
from utils.var_dim import toNumpy
from utils.loader import get_save_path
from utils.var_dim import squeezeToNumpy
from utils.loader import dataLoader_test as dataLoader
from utils.print_tool import datasize
from utils.loader import get_module
from numpy.linalg import norm
from utils.utils import warp_points
from utils.utils import filter_points
from models.model_wrap import PointTracker
from sklearn.metrics import average_precision_score
from evaluations.detector_evaluation import warp_keypoints

def export_descriptor(config, output_dir, args):
   
    
    # basic settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("train on device: %s", device)
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    save_path = get_save_path(output_dir)
    save_output = save_path / "../predictions"
    os.makedirs(save_output, exist_ok=True)
   
    outputMatches = True
    subpixel = config["model"]["subpixel"]["enable"]
    patch_size = config["model"]["subpixel"]["patch_size"]

   
    task = config["data"]["dataset"]  
    data = dataLoader(config, dataset=task)  #读数据图片
    test_set, test_loader = data["test_set"], data["test_loader"]
    datasize(test_loader, config, tag="test")

    ## load pretrained
    Val_model_heatmap = get_module("", config["front_end_model"])
    val_agent = Val_model_heatmap(config["model"], device=device)
    val_agent.loadModel()

    ## tracker
    tracker = PointTracker(max_length=2, nn_thresh=val_agent.nn_thresh)

    ###### check!!!
    count = 0
    for sample in tqdm(test_loader):

        img_0, img_1 = sample["image"], sample["warped_image"]
        img_0 = img_0[0]
        img_1 = img_1[0]
        # first image, no matches
        # img = img_0
        def get_pts_desc_from_agent(val_agent, img):
            """
            pts: list [numpy (3, N)]
            desc: list [numpy (256, N)]
            """
            heatmap_batch = val_agent.run(args.result_path,
                img
            ) 
            pts = val_agent.heatmap_to_pts()
            if subpixel:  
                pts = val_agent.soft_argmax_points(pts, patch_size=patch_size)
            # heatmap, pts to desc
            desc_sparse = val_agent.desc_to_sparseDesc()
            outs = {"pts": pts[0], "desc": desc_sparse[0]}
            return outs

        def transpose_np_dict(outs):
            for entry in list(outs):
                outs[entry] = outs[entry].transpose()

        outs = get_pts_desc_from_agent(val_agent, img_0)
        pts, desc = outs["pts"], outs["desc"]  # pts: np [3, N]

        if outputMatches == True:
            tracker.update(pts, desc)
        # save keypoints
        pred = {"image": squeezeToNumpy(sample["imagetensor"])}    
        pred.update({"prob": pts.transpose(), "desc": desc.transpose()})
        # second image, output matches
        outs = get_pts_desc_from_agent(val_agent, img_1)
        pts, desc = outs["pts"], outs["desc"]
        if outputMatches == True:
            tracker.update(pts, desc)

        pred.update({"warped_image": squeezeToNumpy(sample["warped_imagetensor"])})
        pred.update(
            {
                "warped_prob": pts.transpose(),
                "warped_desc": desc.transpose(),
                "homography": squeezeToNumpy(sample["homography"]),
            }
        )

        if outputMatches == True:
            matches = tracker.get_matches()
            pred.update({"matches": matches.transpose()})

        # clean last descriptor
        tracker.clear_desc()
        filename = str(count)
        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)
        count += 1


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def find_files_with_ext(directory, extension='.npz', if_int=True):
    list_of_files = []
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img

def evaluate(args, **options):
    path = './npz_result/predictions/'
    files = find_files_with_ext(path)
    correctness = []
    est_H_mean_dist = []
    repeatability = []
    mscore = []
    mAP = []
    localization_err = []
    rep_thd = 3
    save_file = path + "/result.txt"
    inliers_method = 'cv'
    compute_map = True
    verbose = True
    top_K = 1000

    reproduce = True
    if reproduce:
        logging.info("reproduce = True")
        np.random.seed(0)

   
    files.sort(key=lambda x: int(x[:-4]))

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + '/' + f)
  
        real_H = data['homography']
        image = data['image']
        warped_image = data['warped_image']
        keypoints = data['prob'][:, [1, 0]]
        warped_keypoints = data['warped_prob'][:, [1, 0]]
   
        if args.homography:
            # estimate result
            ##### check
            homography_thresh = [1,3,5,10,20,50]
            #####
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result['correctness'])
            # est_H_mean_dist.append(result['mean_dist'])
            # compute matching score
            def warpLabels(pnts, homography, H, W):
                pnts = torch.tensor(pnts).long()
                homography = torch.tensor(homography, dtype=torch.float32)
                warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                                          homography)  # check the (x, y)
                warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
                return warped_pnts.numpy()

            #from numpy.linalg import inv
            H, W = image.shape
            unwarped_pnts = warpLabels(warped_keypoints, inv(real_H), H, W)
            score = (result['inliers'].sum() * 2) / (keypoints.shape[0] + unwarped_pnts.shape[0])
            mscore.append(score)
            # compute map
            if compute_map:
                def getMatches(data):

                    desc = data['desc']
                    warped_desc = data['warped_desc']

                    nn_thresh = 1.2
                    #print("nn threshold: ", nn_thresh)
                    tracker = PointTracker(max_length=2, nn_thresh=nn_thresh)
                    # matches = tracker.nn_match_two_way(desc, warped_desc, nn_)
                    tracker.update(keypoints.T, desc.T)
                    tracker.update(warped_keypoints.T, warped_desc.T)
                    matches = tracker.get_matches().T
                    mscores = tracker.get_mscores().T


                    return matches, mscores

                def getInliers(matches, H, epi=3, verbose=False):
                    """
                    input:
                        matches: numpy (n, 4(x1, y1, x2, y2))
                        H (ground truth homography): numpy (3, 3)
                    """
                    # warp points 
                    warped_points = warp_keypoints(matches[:, :2], H) # make sure the input fits the (x,y)

                    # compute point distance
                    norm = np.linalg.norm(warped_points - matches[:, 2:4],
                                            ord=None, axis=1)
                    inliers = norm < epi
                    if verbose:
                        print("Total matches: ", inliers.shape[0], ", inliers: ", inliers.sum(),
                                          ", percentage: ", inliers.sum() / inliers.shape[0])

                    return inliers

                def getInliers_cv(matches, H=None, epi=3, verbose=False):
                    # count inliers: use opencv homography estimation
                    # Estimate the homography between the matches using RANSAC
                    H, inliers = cv2.findHomography(matches[:, [0, 1]],
                                                    matches[:, [2, 3]],
                                                    cv2.RANSAC)
                    inliers = inliers.flatten()
                    return inliers
            
            
                def computeAP(m_test, m_score):

                    average_precision = average_precision_score(m_test, m_score)

                    return average_precision

                def flipArr(arr):
                    return arr.max() - arr
                
                if args.sift:
                    assert result is not None
                    matches, mscores = result['matches'], result['mscores']
                else:
                    matches, mscores = getMatches(data)
                
                real_H = data['homography']
                if inliers_method == 'gt':
                    # use ground truth homography
                    inliers = getInliers(matches, real_H, epi=3, verbose=verbose)
                else:
                    # use opencv estimation as inliers
                    inliers = getInliers_cv(matches, real_H, epi=3, verbose=verbose)
                    
                ## distance to confidence
                if args.sift:
                    m_flip = flipArr(mscores[:])  # for sift
                else:
                    m_flip = flipArr(mscores[:,2])
        
                if inliers.shape[0] > 0 and inliers.sum()>0:
#                     m_flip = flipArr(m_flip)
                    # compute ap
                    ap = computeAP(inliers, m_flip)
                else:
                    ap = 0
                
                mAP.append(ap)


    if args.homography:
        correctness_ave = np.array(correctness).mean(axis=0)
        # est_H_mean_dist = np.array(est_H_mean_dist)
        mscore_m = np.array(mscore).mean(axis=0)
        if compute_map:
            mAP_m = np.array(mAP).mean()
            print("mean AP", mAP_m)

        print("end")



if __name__ == '__main__':
    import argparse


    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="img path")
    parser.add_argument("--result_path", type=str, required=True, help="result path")
    #parser.add_argument("--npz_path", type=str, default="./", help="npz_path")
    parser.add_argument("--sift", type=ast.literal_eval, default=False, help="sift, default is false.")
    parser.add_argument("--outputImg", type=ast.literal_eval, default=True,
                        help="outputImg, default is true.")
    parser.add_argument("--repeatibility", type=ast.literal_eval, default=True,
                        help="repeatibility, default is true.")
    parser.add_argument("--homography", type=ast.literal_eval, default=True,
                        help="homography, default is true.")
    parser.add_argument("--plotMatching", type=ast.literal_eval, default=True,
                        help="plotMatching, default is true.")
    args = parser.parse_args()
    args = parser.parse_args()
    with open(args.img_path, "r") as f:
        config = yaml.safe_load(f)
    output_dir = "./npz_result"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    export_descriptor(config, output_dir, args)
    evaluate(args)
