import os.path as osp
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default="demo/demo.jpg", help='Image file')
    parser.add_argument('--config', default="configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py", help='Config file')
    parser.add_argument('checkpoint', default="work_dirs/fcos_r50_caffe_fpn_4x4_1x_coco/latest.pth", help='Checkpoint file')
    parser.add_argument(
        '--device', default='npu:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='.', help='directory where painted images will be saved')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    if args.show_dir:
        out_file = osp.join(args.show_dir, osp.basename(args.img))
    else:
        out_file = None
    model.show_result(
                    args.img,
                    result,
                    show=args.show,
                    out_file=out_file,
                    score_thr=args.score_thr)


if __name__ == '__main__':
    main()
