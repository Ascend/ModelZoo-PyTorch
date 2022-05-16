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

#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.utils.config import Config

from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

from mmocr.models import build_detector
from mmocr.utils import is_2dlist
from mmocr.apis.utils import disable_text_recog_aug_test
from mmocr.datasets.pipelines.crop import crop_img
from mmocr.models import build_detector
from mmocr.utils.model import revert_sync_batchnorm


# Parse CLI arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'img', type=str, help='Input image file or folder path.')
    parser.add_argument(
        '--output',
        type=str,
        default='',
        help='Output file/folder name for visualization')
    parser.add_argument(
        '--det',
        type=str,
        default='FCE_IC15',
        help='Pretrained text detection algorithm')
    parser.add_argument(
        '--det-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected det model. It '
        'overrides the settings in det')
    parser.add_argument(
        '--det-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected det model. '
        'It overrides the settings in det')
    parser.add_argument(
        '--recog',
        type=str,
        default='None',
        help='Pretrained text recognition algorithm')
    parser.add_argument(
        '--recog-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected recog model. It'
        'overrides the settings in recog')
    parser.add_argument(
        '--recog-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected recog model. '
        'It overrides the settings in recog')
    parser.add_argument(
        '--kie',
        type=str,
        default='',
        help='Pretrained key information extraction algorithm')
    parser.add_argument(
        '--kie-config',
        type=str,
        default='',
        help='Path to the custom config file of the selected kie model. It'
        'overrides the settings in kie')
    parser.add_argument(
        '--kie-ckpt',
        type=str,
        default='',
        help='Path to the custom checkpoint file of the selected kie model. '
        'It overrides the settings in kie')
    parser.add_argument(
        '--config-dir',
        type=str,
        default=os.path.join(str(Path.cwd()), 'configs/'),
        help='Path to the config directory where all the config files '
        'are located. Defaults to "configs/"')
    parser.add_argument(
        '--batch-mode',
        action='store_true',
        help='Whether use batch mode for inference')
    parser.add_argument(
        '--recog-batch-size',
        type=int,
        default=0,
        help='Batch size for text recognition')
    parser.add_argument(
        '--det-batch-size',
        type=int,
        default=0,
        help='Batch size for text detection')
    parser.add_argument(
        '--single-batch-size',
        type=int,
        default=0,
        help='Batch size for separate det/recog inference')
    parser.add_argument(
        '--device', default=None, help='Device used for inference.')
    parser.add_argument(
        '--export',
        type=str,
        default='',
        help='Folder where the results of each image are exported')
    parser.add_argument(
        '--export-format',
        type=str,
        default='json',
        help='Format of the exported result file(s)')
    parser.add_argument(
        '--details',
        action='store_true',
        help='Whether include the text boxes coordinates and confidence values'
    )
    parser.add_argument(
        '--imshow',
        action='store_true',
        help='Whether show image with OpenCV.')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Prints the recognised text')
    parser.add_argument(
        '--merge', action='store_true', help='Merge neighboring boxes')
    parser.add_argument(
        '--merge-xdist',
        type=float,
        default=20,
        help='The maximum x-axis distance to merge boxes')
    args = parser.parse_args()
    if args.det == 'None':
        args.det = None
    if args.recog == 'None':
        args.recog = None
    # Warnings
    if args.merge and not (args.det and args.recog):
        warnings.warn(
            'Box merging will not work if the script is not'
            ' running in detection + recognition mode.', UserWarning)
    if not os.path.samefile(args.config_dir, os.path.join(str(
            Path.cwd()))) and (args.det_config != ''
                               or args.recog_config != ''):
        warnings.warn(
            'config_dir will be overridden by det-config or recog-config.',
            UserWarning)
    return args


class MMOCR:

    def __init__(self,
                 det='FCE_IC15',
                 det_config='',
                 det_ckpt='',
                 recog='None',
                 recog_config='',
                 recog_ckpt='',
                 kie='',
                 kie_config='',
                 kie_ckpt='',
                 config_dir=os.path.join(str(Path.cwd()), 'configs/'),
                 device=None,
                 **kwargs):

        textdet_models = {
            'DB_r18': {
                'config':
                'dbnet/dbnet_r18_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth'
            },
            'DB_r50': {
                'config':
                'dbnet/dbnet_r50dcnv2_fpnc_1200e_icdar2015.py',
                'ckpt':
                'dbnet/'
                'dbnet_r50dcnv2_fpnc_sbn_1200e_icdar2015_20211025-9fe3b590.pth'
            },
            'DRRG': {
                'config':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'drrg/drrg_r50_fpn_unet_1200e_ctw1500_20211022-fb30b001.pth'
            },
            'FCE_IC15': {
                'config':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015.py',
                'ckpt':
                'fcenet/fcenet_r50_fpn_1500e_icdar2015_20211022-daefb6ed.pth'
            },
            'FCE_CTW_DCNv2': {
                'config':
                'fcenet/fcenet_r50dcnv2_fpn_1500e_ctw1500.py',
                'ckpt':
                'fcenet/' +
                'fcenet_r50dcnv2_fpn_1500e_ctw1500_20211022-e326d7ec.pth'
            },
            'MaskRCNN_CTW': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_ctw1500.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_ctw1500_20210219-96497a76.pth'
            },
            'MaskRCNN_IC15': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2015.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2015_20210219-8eb340a3.pth'
            },
            'MaskRCNN_IC17': {
                'config':
                'maskrcnn/mask_rcnn_r50_fpn_160e_icdar2017.py',
                'ckpt':
                'maskrcnn/'
                'mask_rcnn_r50_fpn_160e_icdar2017_20210218-c6ec3ebb.pth'
            },
            'PANet_CTW': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_ctw1500.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_ctw1500_20210219-3b3a9aa3.pth'
            },
            'PANet_IC15': {
                'config':
                'panet/panet_r18_fpem_ffm_600e_icdar2015.py',
                'ckpt':
                'panet/'
                'panet_r18_fpem_ffm_sbn_600e_icdar2015_20210219-42dbe46a.pth'
            },
            'PS_CTW': {
                'config': 'psenet/psenet_r50_fpnf_600e_ctw1500.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_ctw1500_20210401-216fed50.pth'
            },
            'PS_IC15': {
                'config':
                'psenet/psenet_r50_fpnf_600e_icdar2015.py',
                'ckpt':
                'psenet/psenet_r50_fpnf_600e_icdar2015_pretrain-eefd8fe6.pth'
            },
            'TextSnake': {
                'config':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py',
                'ckpt':
                'textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth'
            }
        }

        textrecog_models = {
            'CRNN': {
                'config': 'crnn/crnn_academic_dataset.py',
                'ckpt': 'crnn/crnn_academic-a723a1c5.pth'
            },
            'SAR': {
                'config': 'sar/sar_r31_parallel_decoder_academic.py',
                'ckpt': 'sar/sar_r31_parallel_decoder_academic-dba3a4a3.pth'
            },
            'SAR_CN': {
                'config':
                'sar/sar_r31_parallel_decoder_chinese.py',
                'ckpt':
                'sar/sar_r31_parallel_decoder_chineseocr_20210507-b4be8214.pth'
            },
            'NRTR_1/16-1/8': {
                'config': 'nrtr/nrtr_r31_1by16_1by8_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by16_1by8_academic_20211124-f60cebf4.pth'
            },
            'NRTR_1/8-1/4': {
                'config': 'nrtr/nrtr_r31_1by8_1by4_academic.py',
                'ckpt':
                'nrtr/nrtr_r31_1by8_1by4_academic_20211123-e1fdb322.pth'
            },
            'RobustScanner': {
                'config': 'robust_scanner/robustscanner_r31_academic.py',
                'ckpt': 'robustscanner/robustscanner_r31_academic-5f05874f.pth'
            },
            'SATRN': {
                'config': 'satrn/satrn_academic.py',
                'ckpt': 'satrn/satrn_academic_20211009-cb8b1580.pth'
            },
            'SATRN_sm': {
                'config': 'satrn/satrn_small.py',
                'ckpt': 'satrn/satrn_small_20211009-2cf13355.pth'
            },
            'ABINet': {
                'config': 'abinet/abinet_academic.py',
                'ckpt': 'abinet/abinet_academic-f718abf6.pth'
            },
            'SEG': {
                'config': 'seg/seg_r31_1by16_fpnocr_academic.py',
                'ckpt': 'seg/seg_r31_1by16_fpnocr_academic-72235b11.pth'
            },
            'CRNN_TPS': {
                'config': 'tps/crnn_tps_academic_dataset.py',
                'ckpt': 'tps/crnn_tps_academic_dataset_20210510-d221a905.pth'
            }
        }

        kie_models = {
            'SDMGR': {
                'config': 'sdmgr/sdmgr_unet16_60e_wildreceipt.py',
                'ckpt':
                'sdmgr/sdmgr_unet16_60e_wildreceipt_20210520-7489e6de.pth'
            }
        }

        self.td = det
        self.tr = recog
        self.kie = kie
        self.device = device
        if self.device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        # Check if the det/recog model choice is valid
        if self.td and self.td not in textdet_models:
            raise ValueError(self.td,
                             'is not a supported text detection algorthm')
        elif self.tr and self.tr not in textrecog_models:
            raise ValueError(self.tr,
                             'is not a supported text recognition algorithm')
        elif self.kie:
            if self.kie not in kie_models:
                raise ValueError(
                    self.kie, 'is not a supported key information extraction'
                    ' algorithm')
            elif not (self.td and self.tr):
                raise NotImplementedError(
                    self.kie, 'has to run together'
                    ' with text detection and recognition algorithms.')

        self.detect_model = None
        if self.td:
            # Build detection model
            if not det_config:
                det_config = os.path.join(config_dir, 'textdet/',
                                          textdet_models[self.td]['config'])
            if not det_ckpt:
                det_ckpt = 'https://download.openmmlab.com/mmocr/textdet/' + \
                    textdet_models[self.td]['ckpt']

            self.detect_model = init_detector(
                det_config, det_ckpt, device=self.device)
            self.detect_model = revert_sync_batchnorm(self.detect_model)

        self.recog_model = None
        if self.tr:
            # Build recognition model
            if not recog_config:
                recog_config = os.path.join(
                    config_dir, 'textrecog/',
                    textrecog_models[self.tr]['config'])
            if not recog_ckpt:
                recog_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'textrecog/' + textrecog_models[self.tr]['ckpt']

            self.recog_model = init_detector(
                recog_config, recog_ckpt, device=self.device)
            self.recog_model = revert_sync_batchnorm(self.recog_model)

        self.kie_model = None
        if self.kie:
            # Build key information extraction model
            if not kie_config:
                kie_config = os.path.join(config_dir, 'kie/',
                                          kie_models[self.kie]['config'])
            if not kie_ckpt:
                kie_ckpt = 'https://download.openmmlab.com/mmocr/' + \
                    'kie/' + kie_models[self.kie]['ckpt']

            kie_cfg = Config.fromfile(kie_config)
            self.kie_model = build_detector(
                kie_cfg.model, test_cfg=kie_cfg.get('test_cfg'))
            self.kie_model = revert_sync_batchnorm(self.kie_model)
            self.kie_model.cfg = kie_cfg
            load_checkpoint(self.kie_model, kie_ckpt, map_location=self.device)

        # Attribute check
        for model in list(filter(None, [self.recog_model, self.detect_model])):
            if hasattr(model, 'module'):
                model = model.module

    def readtext(self,
                 img,
                 output=None,
                 details=False,
                 export=None,
                 export_format='json',
                 batch_mode=False,
                 recog_batch_size=0,
                 det_batch_size=0,
                 single_batch_size=0,
                 imshow=False,
                 print_result=False,
                 merge=False,
                 merge_xdist=20,
                 **kwargs):
        args = locals().copy()
        [args.pop(x, None) for x in ['kwargs', 'self']]
        args = Namespace(**args)
        # Input and output arguments processing
        self._args_processing(args)
        self.args = args
        pp_result = None
        # Send args and models to the MMOCR model inference API
        # and call post-processing functions for the output
        if self.detect_model and self.recog_model:
            det_recog_result = self.det_recog_kie_inference(
                self.detect_model, self.recog_model, kie_model=self.kie_model)
            pp_result = self.det_recog_pp(det_recog_result)
        else:
            for model in list(
                    filter(None, [self.recog_model, self.detect_model])):
                self.single_inference(model, args.filenames, args.arrays,
                                               args.batch_mode,
                                               args.single_batch_size)

    # Separate det/recog inference pipeline
    def single_inference(self, model, filenames, arrays, batch_mode, batch_size=0):
        result = []
        if batch_mode:
            if batch_size == 0:
                result = model_inference(model, arrays, filenames, batch_mode=True)
            else:
                n = batch_size
                arr_chunks = [
                    arrays[i:i + n] for i in range(0, len(arrays), n)
                ]
                for chunk in arr_chunks:
                    result.extend(
                        model_inference( model, chunk, filenames, batch_mode=True))
        else:
            for (arr,filename) in zip(arrays,filenames) :
                model_inference(model, arr, filename, batch_mode=False)

    # Arguments pre-processing function
    def _args_processing(self, args):
        # Check if the input is a list/tuple that
        # contains only np arrays or strings
        if isinstance(args.img, (list, tuple)):
            img_list = args.img
            if not all([isinstance(x, (np.ndarray, str)) for x in args.img]):
                raise AssertionError('Images must be strings or numpy arrays')
        # Create a list of the images
        if isinstance(args.img, str):
            img_path = Path(args.img)
            if img_path.is_dir():
                img_list = [str(x) for x in img_path.glob('*')]
            else:
                img_list = [str(img_path)]
        elif isinstance(args.img, np.ndarray):
            img_list = [args.img]
        # Read all image(s) in advance to reduce wasted time
        # re-reading the images for visualization output
        args.arrays = [mmcv.imread(x) for x in img_list]

        # Create a list of filenames (used for output images and result files)
        if isinstance(img_list[0], str):
            args.filenames = [str(Path(x).stem) for x in img_list]
        else:
            args.filenames = [str(x) for x in range(len(img_list))]

        # If given an output argument, create a list of output image filenames
        num_res = len(img_list)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir():
                args.output = [
                    str(output_path / f'out_{x}.png') for x in args.filenames
                ]
            else:
                args.output = [str(args.output)]
                if args.batch_mode:
                    raise AssertionError('Output of multiple images inference'
                                         ' must be a directory')
        else:
            args.output = [None] * num_res

        # If given an export argument, create a list of
        # result filenames for each image
        if args.export:
            export_path = Path(args.export)
            args.export = [
                str(export_path / f'out_{x}.{args.export_format}')
                for x in args.filenames
            ]
        else:
            args.export = [None] * num_res

        return args

def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if config.model.get('pretrained'):
        config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

def model_inference(model,
                    imgs,
                    filename,
                    ann=None,
                    batch_mode=False,
                    return_data=False):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
            Either image files or loaded images.
        batch_mode (bool): If True, use batch mode for inference.
        ann (dict): Annotation info for key information extraction.
        return_data: Return postprocessed data.
    Returns:
        result (dict): Predicted results.
    """
    if isinstance(imgs, (list, tuple)):
        is_batch = True
        if len(imgs) == 0:
            raise Exception('empty imgs provided, please check and try again')
        if not isinstance(imgs[0], (np.ndarray, str)):
            raise AssertionError('imgs must be strings or numpy arrays')

    elif isinstance(imgs, (np.ndarray, str)):
        imgs = [imgs]
        is_batch = False
    else:
        raise AssertionError('imgs must be strings or numpy arrays')

    is_ndarray = isinstance(imgs[0], np.ndarray)
    cfg = model.cfg

    if batch_mode:
        cfg = disable_text_recog_aug_test(cfg, set_types=['test'])

    device = next(model.parameters()).device  # model device

    if cfg.data.test.get('pipeline', None) is None:
        if is_2dlist(cfg.data.test.datasets):
            cfg.data.test.pipeline = cfg.data.test.datasets[0][0].pipeline
        else:
            cfg.data.test.pipeline = cfg.data.test.datasets[0].pipeline
    if is_2dlist(cfg.data.test.pipeline):
        cfg.data.test.pipeline = cfg.data.test.pipeline[0]

    if is_ndarray:
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromNdarray'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    for img in imgs:
        # prepare data
        if is_ndarray:
            # directly add img
            data = dict(
                img=img,
                ann_info=ann,
                img_info=dict(width=img.shape[1], height=img.shape[0]),
                bbox_fields=[])
        else:
            # add information into dict
            data = dict(
                img_info=dict(filename=img),
                img_prefix=None,
                ann_info=ann,
                bbox_fields=[])
        if ann is not None:
            data.update(dict(**ann))
        data = test_pipeline(data)

        # get tensor from list to stack for batch mode (text detection)
        if batch_mode:
            if cfg.data.test.pipeline[1].type == 'MultiScaleFlipAug':
                for key, value in data.items():
                    data[key] = value[0]
        datas.append(data)
    if isinstance(datas[0]['img'], list) and len(datas) > 1:
        raise Exception('aug test does not support '
                        f'inference with batch size '
                        f'{len(datas)}')

    data = collate(datas, samples_per_gpu=len(imgs))
    
    # process img_metas
    if isinstance(data['img_metas'], list):
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
    else:
        data['img_metas'] = data['img_metas'].data

    if isinstance(data['img'], list):
        data['img'] = [img.data for img in data['img']]
        if isinstance(data['img'][0], list):
            data['img'] = [img[0] for img in data['img']]
    else:
        data['img'] = data['img'].data

    # for KIE models
    if ann is not None:
        data['relations'] = data['relations'].data[0]
        data['gt_bboxes'] = data['gt_bboxes'].data[0]
        data['texts'] = data['texts'].data[0]
        data['img'] = data['img'][0]
        data['img_metas'] = data['img_metas'][0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
        
    
    #input bin file
    save_path = "/home/zhangyifan/icdar2015/preprocessed_imgs_test_new/test"
    input_tensor = data['img'][0][0]
    img = np.array(input_tensor).astype(np.float32)
    img.tofile(os.path.join(save_path, filename + ".bin"))

# Create an inference pipeline with parsed arguments
def main():
    args = parse_args()
    ocr = MMOCR(**vars(args))
    ocr.readtext(**vars(args))


if __name__ == '__main__':
    main()
