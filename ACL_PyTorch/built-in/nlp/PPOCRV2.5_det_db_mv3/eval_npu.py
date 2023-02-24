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


from tqdm import tqdm
from ppocr.data import build_dataloader
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
import tools.program as program
import paddle
import numpy as np
from ais_bench.infer.interface import InferSession


def eval(session_dbnet,
         valid_dataloader,
         post_process_class,
         eval_class):
    with paddle.no_grad():
        pbar = tqdm(
            total=len(valid_dataloader) // 24,
            desc='eval model:',
            position=0,
            leave=True)
        temp = []
        result_array = []
        for idx, batch in enumerate(valid_dataloader):
            temp_item = []
            for item in batch:
                if isinstance(item, paddle.Tensor):
                    temp_item.append(item.numpy())
                else:
                    temp_item.append(item)
            temp.append(temp_item)
            if len(temp) == 24:
                result_array.append(temp)
                temp = []
        
        for images_bstch in  result_array:
            temp_imgs = []
            for imge, _, _, _ in images_bstch:
                temp_imgs.append(imge)
            
            imges = np.concatenate(temp_imgs, axis=0)
            outputs = session_dbnet.infer([imges])[0]
            
            for i in range(24):
                preds = outputs[i,:,:,:][np.newaxis,:]
                preds = {'maps': preds}
            # Evaluate the results of the current batch
                post_result = post_process_class(preds, images_bstch[i][1])
                eval_class(post_result, images_bstch[i])
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

    # om
    device_id = global_config['device_id']
    om_path = global_config['om_path']
    session_dbnet = InferSession(device_id=device_id, model_path=om_path)
   
    # build metric
    eval_class = build_metric(config['Metric'])

    # start eval
    metric = eval(session_dbnet, valid_dataloader, post_process_class,
                          eval_class)

    total_time = 0
    s = session_dbnet.sumary()
    total_time += np.mean(s.exec_time_list)
    
    metric['fps'] = 1000 * 24 / total_time
    logger.info('metric eval ***************')
    for k, v in metric.items():
        logger.info('{}:{}'.format(k, v))


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
