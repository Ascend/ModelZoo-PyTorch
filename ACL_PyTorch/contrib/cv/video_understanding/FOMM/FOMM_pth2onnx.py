import torch
from utils import load_checkpoints
from argparse import ArgumentParser
from torch import onnx


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--outdir", default='onnxs', help="dir where onnx files to save")
    parser.add_argument("--genname", required=True, help="name of the generator.onnx model file")
    parser.add_argument("--kpname", required=True, help="name of the kp-detector.onnx model file")

    opt = parser.parse_args()
    generator, kp_detector = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint)
    data_name = opt.checkpoint.split('/')[1].split('-')[0]
    if opt.checkpoint.split('/')[1].__contains__('adv'):
        data_name += '-adv'
    source_imgs = torch.randn(1, 3, 256, 256)
    kp_driving = {"value": torch.randn(1, 10, 2), "jacobian": torch.randn(1, 10, 2, 2)}
    kp_source = {"value": torch.randn(1, 10, 2), "jacobian": torch.randn(1, 10, 2, 2)}
    gen_dummy_input = (source_imgs, {"kp_driving": kp_driving, "kp_source": kp_source})
    kpdet_dummy_input = torch.randn(1, 3, 256, 256)

    gen_save_path = opt.outdir + "/" + opt.genname + ".onnx"
    kpdet_save_path = opt.outdir + "/" + opt.kpname + ".onnx"
    gen_input_names = ["source_imgs", "kp_driving_value", "kp_driving_jac", "kp_source_value", "kp_source_jac"]
    output_names = ["output"]
    kp_input_names = ["input"]

    torch.onnx.export(kp_detector, kpdet_dummy_input, kpdet_save_path, input_names=kp_input_names,
                      output_names=output_names, opset_version=11)
    torch.onnx.export(generator, gen_dummy_input, gen_save_path, input_names=gen_input_names,
                      output_names=output_names, opset_version=11)
