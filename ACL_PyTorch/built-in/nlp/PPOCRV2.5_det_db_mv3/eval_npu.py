# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from tqdm import tqdm
import paddle
import numpy as np
from ais_bench.infer.interface import InferSession, MemorySummary

from PaddleOCR.tools import program
from PaddleOCR.ppocr.data import build_dataloader
from PaddleOCR.ppocr.postprocess import build_post_process
from PaddleOCR.ppocr.metrics import build_metric


def eval(session_dbnet,
         valid_dataloader,
         post_process_class,
         eval_class,
         save_npu_path,
         batch_size):
    div, mod = divmod(len(valid_dataloader),batch_size)
    if mod == 0:
        total_epoch = div
    else:
        total_epoch = div + 1
    with paddle.no_grad():
        pbar = tqdm(
            total=total_epoch,
            desc='eval model:',
            position=0,
            leave=True)
        temp = []
        result_array = []
        shape_ = None
        for batch in valid_dataloader:
            temp_item = []
            for ind, item in enumerate(batch):
                if isinstance(item, paddle.Tensor):
                    if ind == 0:
                        # NCHW ----> CHW, N==1
                        ig = item.numpy()[0]
                        # CHW ----> HWC
                        ig = ig.transpose((1,2,0))
                        ig = np.ascontiguousarray(ig)
                        shape_ = ig.shape
                        ig = ig[np.newaxis,:].astype(np.uint8)
                        temp_item.append(ig)
                    else:
                        temp_item.append(item.numpy())
                else:
                    temp_item.append(item)
            temp.append(temp_item)
            if len(temp) == batch_size:
                result_array.append(temp)
                temp = []

        zeros = np.zeros((1,*shape_),dtype=np.uint8)
        while len(temp) != batch_size:
            temp.append((zeros,'pass','',''))
        result_array.append(temp)

        for ind, images_batch in  enumerate(result_array):
            temp_imgs = []
            for imge, _, _, _ in images_batch:
                temp_imgs.append(imge)
            
            imges = np.concatenate(temp_imgs, axis=0)
            outputs = session_dbnet.infer([imges])[0]
            
            for i in range(batch_size):
                preds = outputs[i,:,:,:][np.newaxis,:]
                
                preds = {'maps': preds}
            # Evaluate the results of the current batch
                if images_batch[i][1] == 'pass':
                    continue
                np.save(f'{save_npu_path}/{ind}_{i}.npy', preds)
                post_result = post_process_class(preds, images_batch[i][1])
                eval_class(post_result, images_batch[i])
            pbar.update(1)
           
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    pbar.close()
  
    return metric



def main():
    global_config = config['Global']
    # build dataloader
    valid_dataloader = build_dataloader(config, 'Eval', device, logger)

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)
    
    # bs
    batch_size = int(global_config['batch_size'])
    # om
    device_id = global_config['device_id']
    om_path = global_config['om_path']
    session_dbnet = InferSession(device_id=device_id, model_path=om_path)
   
    # build metric
    eval_class = build_metric(config['Metric'])

    save_npu_path = global_config['save_npu_path']
    if not os.path.exists(save_npu_path):
        os.mkdir(save_npu_path)
    # start eval
    metric = eval(session_dbnet, valid_dataloader, post_process_class,
                          eval_class, save_npu_path,batch_size)

    s = session_dbnet.sumary()

    metric['fps_without_d2h_h2d'] = 1000 * batch_size / np.mean(s.exec_time_list)
    metric['fps_with_d2h_h2d'] = 1000 * batch_size / (np.mean(s.exec_time_list) + \
        np.mean(MemorySummary.get_H2D_time_list()) + \
            np.mean(MemorySummary.get_D2H_time_list()))
    metric['h2d(ms)'] = np.mean(MemorySummary.get_H2D_time_list())
    metric['d2h(ms)'] = np.mean(MemorySummary.get_D2H_time_list())

    logger.info('metric eval ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
