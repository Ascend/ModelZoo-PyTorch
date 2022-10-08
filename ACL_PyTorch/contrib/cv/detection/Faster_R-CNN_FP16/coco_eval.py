import argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def coco_evaluation(annotation_json, result_json):
    cocoGt = COCO(annotation_json)
    cocoDt = cocoGt.loadRes(result_json)
    iou_thrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    iou_type = 'bbox'

    cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
    cocoEval.params.catIds = cocoGt.get_cat_ids(cat_names=CLASSES)
    cocoEval.params.imgIds = cocoGt.get_img_ids()
    cocoEval.params.maxDets = [100, 300, 1000] # proposal number for evaluating recalls/mAPs.
    cocoEval.params.iouThrs = iou_thrs

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # mapping of cocoEval.stats
    coco_metric_names = {
        'mAP': 0,
        'mAP_50': 1,
        'mAP_75': 2,
        'mAP_s': 3,
        'mAP_m': 4,
        'mAP_l': 5,
        'AR@100': 6,
        'AR@300': 7,
        'AR@1000': 8,
        'AR_s@1000': 9,
        'AR_m@1000': 10,
        'AR_l@1000': 11
    }

    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    eval_results = {}

    for metric_item in metric_items:
        key = f'bbox_{metric_item}'
        val = float(
            f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
        )
        eval_results[key] = val
    ap = cocoEval.stats[:6]
    eval_results['bbox_mAP_copypaste'] = (
        f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
        f'{ap[4]:.3f} {ap[5]:.3f}')
    
    return eval_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth", default="instances_val2017.json")
    parser.add_argument("--detection_result", default="coco_detection_result.json")
    args = parser.parse_args()
    result = coco_evaluation(args.ground_truth, args.detection_result)
    print(result)
    with open('./coco_detection_result.txt', 'w') as f:
        for key, value in result.items():
            f.write(key + ': ' + str(value) + '\n')