# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bowen Cheng (bcheng9@illinois.edu) and Bin Xiao (leoxiaobin@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path
from pathlib import Path

import cv2
import json_tricks as json
import numpy as np
from torch.utils.data import Dataset
import pickle

from crowdposetools.cocoeval import COCOeval
from dataset.JointsLambdaDataset import JointsLambdaDataset
from nms.nms import oks_nms

from utils import zipreader

logger = logging.getLogger(__name__)

crowdpose_sigmas = np.array([.79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89, .25, .25]) / 10.0

class COCODataset(JointsLambdaDataset):
    """`CrowdPose`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, cfg, image_dir, annotation_file, dataset_type, image_set, is_train, transform=None):
        super().__init__(cfg, image_dir, annotation_file, image_set, is_train, transform)



    # def __init__(self, root, dataset, data_format, transform=None,
    #              target_transform=None):
        from crowdposetools.coco import COCO
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.scale_thre = cfg.TEST.SCALE_THRE


        self.dataset_type = dataset_type
        self.coco = COCO(self._get_anno_file_keypoint())

        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 14
        self.flip_pairs = [[0, 1], [2, 3], [4, 5], [6, 7],
                           [8, 9], [10, 11]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 12, 13)
        self.lower_body_ids = (6, 7, 8, 9, 10, 11)

        self.joints_weight = np.array(
            [
                1., 1., 1.2, 1.2,
                1.5, 1.5, 1., 1., 
                1.2, 1.2, 1.5, 1.5,
                1., 1.
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

        return

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_anno_file_keypoint(self):
        # example: root/json/crowdpose_{train,val,test}.json
        return self.annotation_file

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        cache_dir = os.path.join('cache', self.dataset_type, str(self.image_set))
        cache_file = os.path.join(cache_dir, 'gt_db.pkl')

        if not os.path.exists(cache_dir):
            make_cache_dir = Path(cache_dir)
            make_cache_dir.mkdir(parents=True)
        
        if not os.path.exists(cache_file):
            gt_db = []
            
            for index in self.image_set_index:
                gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
            
            print('storing as cache at {}'.format(cache_file))
            with open(cache_file, 'wb') as f:
                pickle.dump(gt_db, f)
            print('done!')

        else:
            print('loading from cache from {}'.format(cache_file))
            with open(cache_file, 'rb') as f:
                gt_db = pickle.load(f)
            print('done!')
            
        return gt_db

    def _get_image_path(self, file_name):
        return os.path.join(self.image_dir, file_name)

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                center, scale = self._box2cs(obj['clean_bbox'][:4])
                obj['center'] = center

                if 'original_annotation_id' not in obj.keys():
                    obj['original_annotation_id'] = obj['id']

                valid_objs.append(obj)
        
        objs = valid_objs

        ## ---- now compute nn to each bounding box with respect to center------
        for idx, obj in enumerate(objs):
            min_dist = 1e10
            min_dist_idx = None

            for other_idx, other_obj in enumerate(objs):
                if other_idx == idx:
                    continue

                dist = ((obj['center'] - other_obj['center'])**2).sum()

                if dist < min_dist:
                    min_dist = dist
                    min_dist_idx = other_idx

            obj['min_dist_annotation_idx'] = min_dist_idx

        #--------------------
        rec = []
        for idx, obj in enumerate(objs):
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            # ---------------------
            image_file_name = im_ann['file_name'].split('/')[-1]
            image_path = os.path.join(self.image_dir, image_file_name)

            # ---------------------------------------
            min_dist_annotation_idx = obj['min_dist_annotation_idx']

            if min_dist_annotation_idx is not None:
                min_dist_annotation = objs[min_dist_annotation_idx]
                min_dist_db_rec = self._only_load_coco_keypoint_annotation(min_dist_annotation, im_ann)[0]
            else:
                min_dist_db_rec = None

            obj['area'] = scale[0]*scale[1]*200*200
            rec.append({
                'image': image_path,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': index,
                'annotation_id': obj['id'],
                'num_keypoints': obj['num_keypoints'],
                'area': obj['area'],
                'original_annotation_id': obj['original_annotation_id'],
                'min_dist_db_rec': min_dist_db_rec
            })

        return rec

    ## used for loading only max overlap annotations
    def _only_load_coco_keypoint_annotation(self, obj, im_ann):
        objs = [obj]

        width = im_ann['width']
        height = im_ann['height']
        
        rec = []
        for idx, obj in enumerate(objs):
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])

            # ---------------------
            image_name = im_ann['file_name'].split('/')[-1]
            image_path = os.path.join(self.image_dir, image_name)

            obj['area'] = scale[0]*scale[1]*200*200
            rec.append({
                'image': image_path,
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                'annotation_id': obj['id'],
                'num_keypoints': obj['num_keypoints'],
                'area': obj['area'],
                'original_annotation_id': obj['original_annotation_id'],
            })

        assert(len(rec) == 1)

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            # scale = scale * 1.25
            scale = scale * self.scale_thre

        return center, scale


    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])

            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after filter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db
    

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def processKeypoints(self, keypoints):
        tmp = keypoints.copy()
        if keypoints[:, 2].max() > 0:
            p = keypoints[keypoints[:, 2] > 0][:, :2].mean(axis=0)
            num_keypoints = keypoints.shape[0]
            for i in range(num_keypoints):
                tmp[i][0:3] = [
                    float(keypoints[i][0]),
                    float(keypoints[i][1]),
                    float(keypoints[i][2])
                ]

        return tmp

    # def evaluate(self, cfg, preds, scores, output_dir,
    #              *args, **kwargs):
    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, epoch=-1,
                 *args, **kwargs):
        '''
        Perform evaluation on COCO keypoint task
        :param cfg: cfg dictionary
        :param preds: prediction
        :param output_dir: output directory
        :param args: 
        :param kwargs: 
        :return: 
        '''
        if all_boxes.shape[1] == 8:
            return self.evaluate_lambda(cfg, preds, output_dir, all_boxes, img_path, epoch, *args, **kwargs)

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))
        
        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_epoch{}.json'.format(
                self.image_set, epoch)
        )

        image_path_to_image_id = {}

        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            img_path_key = os.path.join(self.image_dir, im_ann['file_name'])
            image_path_to_image_id[img_path_key] = im_ann['id']

        # preds is a numpy array: person x (keypoints): N x 14 x 3
        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
            kpt = self.processKeypoints(kpt)

            _kpts.append({
                'keypoints': kpt[:, 0:3],
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': image_path_to_image_id[img_path[idx]],
                'annotation_id': int(all_boxes[idx][6]),
            })

        # keypoints: num_joints * 4 (x, y, score, tag)
        # image x person x (keypoints)
        kpts = defaultdict(list)

        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        oks_nmsed_kpts = []
        # image x person x (keypoints)
        for img in kpts.keys():
            # person x (keypoints)
            img_kpts = kpts[img]
            # person x (keypoints)
            # do not use nms, keep all detections
            keep = []
            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file
        )

        # CrowdPose `test` set has annotation.
        info_str = self._do_python_keypoint_eval(
            res_file, res_folder
        )
        name_value = OrderedDict(info_str)
        return name_value, name_value['AP']

    # --------------------------------------------------------------------
    def evaluate_lambda(self, cfg, preds, output_dir, all_boxes, img_path, epoch=-1,
                 *args, **kwargs):

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_epoch{}.json'.format(
                self.image_set, epoch)
        )

        res_file_mode0 = os.path.join(
            res_folder, 'keypoints_{}_results_mode0_epoch{}.json'.format(
                self.image_set, epoch)
        )

        res_file_mode1 = os.path.join(
            res_folder, 'keypoints_{}_results_mode1_epoch{}.json'.format(
                self.image_set, epoch)
        )

        image_path_to_image_id = {}

        for index in self.image_set_index:
            im_ann = self.coco.loadImgs(index)[0]
            img_path_key = os.path.join(self.image_dir, im_ann['file_name'])
            image_path_to_image_id[img_path_key] = im_ann['id']

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            area = (np.max(kpt[:, 0]) - np.min(kpt[:, 0])) * (np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
            kpt = self.processKeypoints(kpt)

            _kpts.append({
                'keypoints': kpt[:, 0:3],
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': image_path_to_image_id[img_path[idx]],
                'annotation_id': int(all_boxes[idx][6]),
                'mode': int(all_boxes[idx][7])
            })

        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        oks_nmsed_kpts_mode0 = []
        oks_nmsed_kpts_mode1 = []

        before_len_kps = 0
        for img in kpts:
            img_kpts = kpts[img]
            before_len_kps += len(img_kpts) 

        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score
                n_p['box_score'] = box_score
                n_p['keypoint_score'] = kpt_score

            img_kpts_mode0 = [img_kpts[i] for i in range(len(img_kpts)) if img_kpts[i]['mode'] == 0]
            img_kpts_mode1 = [img_kpts[i] for i in range(len(img_kpts)) if img_kpts[i]['mode'] == 1]

            # # # ------------------------------
            # keep_mode0 = oks_nms(img_kpts_mode0, oks_thre)
            # keep_mode1 = oks_nms(img_kpts_mode1, oks_thre)
            # keep = oks_nms(img_kpts, oks_thre)
            
            # oks_img_kpts_mode0 = [img_kpts_mode0[_keep] for _keep in keep_mode0]
            # oks_img_kpts_mode1 = [img_kpts_mode1[_keep] for _keep in keep_mode1]

            # oks_img_kpts_merged = oks_img_kpts_mode0 + oks_img_kpts_mode1
            # # oks_img_kpts_merged = oks_merge(kpts_db_mode0=img_kpts_mode0, kpts_db_mode1=img_kpts_mode1, min_oks_thres=0.95)

            # # ------------------------------
            img_kpts_merged = img_kpts_mode0 + img_kpts_mode1
            keep = oks_nms(img_kpts_merged, oks_thre)
            oks_img_kpts_merged = [img_kpts_merged[_keep] for _keep in keep]

            keep_mode0 = oks_nms(img_kpts_mode0, oks_thre)
            oks_img_kpts_mode0 = [img_kpts_mode0[_keep] for _keep in keep_mode0]

            keep_mode1 = oks_nms(img_kpts_mode1, oks_thre)
            oks_img_kpts_mode1 = [img_kpts_mode1[_keep] for _keep in keep_mode1]


            # ------------------------------
            if len(keep_mode0) == 0:
                oks_nmsed_kpts_mode0.append(img_kpts_mode0)
            else:
                oks_nmsed_kpts_mode0.append(oks_img_kpts_mode0)

            if len(keep_mode1) == 0:
                oks_nmsed_kpts_mode1.append(img_kpts_mode1)
            else:
                oks_nmsed_kpts_mode1.append(oks_img_kpts_mode1)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append(oks_img_kpts_merged)

        after_len_kps = 0
        for temp in oks_nmsed_kpts:
            after_len_kps += len(temp) 
        print('before #kps:{}, after #kps:{}'.format(before_len_kps, after_len_kps))

        ##------------------------------
        self._write_coco_keypoint_results(oks_nmsed_kpts_mode0, res_file_mode0)
        self._write_coco_keypoint_results(oks_nmsed_kpts_mode1, res_file_mode1)
        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file) ## merged

        ##------------------------------
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(res_file, res_folder)
            name_value = OrderedDict(info_str)

            info_str_mode0 = self._do_python_keypoint_eval(res_file_mode0, res_folder)
            name_value_mode0 = OrderedDict(info_str_mode0)

            info_str_mode1 = self._do_python_keypoint_eval(res_file_mode1, res_folder)
            name_value_mode1 = OrderedDict(info_str_mode1)

            return name_value, name_value_mode0, name_value_mode1, name_value['AP']
        else:
            return {'Null': 0}, {'Null': 0}, {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []
        num_joints = 14

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]['keypoints'] for k in range(len(img_kpts))]
            )
            key_points = np.zeros(
                (_key_points.shape[0], num_joints * 3),
                dtype=np.float
            )

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            for k in range(len(img_kpts)):
                kpt = key_points[k].reshape((num_joints, 3))
                left_top = np.amin(kpt, axis=0)
                right_bottom = np.amax(kpt, axis=0)

                w = right_bottom[0] - left_top[0]
                h = right_bottom[1] - left_top[1]

                cat_results.append({
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'bbox': list([left_top[0], left_top[1], w, h])
                })

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP (easy)', 'AP (medium)', 'AP (hard)']
        stats_index = [0, 1, 2, 5, 6, 7, 8, 9, 10]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[stats_index[ind]]))
            # info_str.append(coco_eval.stats[ind])

        return info_str