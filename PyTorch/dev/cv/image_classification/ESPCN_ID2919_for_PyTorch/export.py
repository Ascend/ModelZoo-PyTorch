#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ============================================================================
#
# Convert Trained Model to ONNX/CoreML/TF/TFLite/TF.js
import os
import logging as log
from argparse import ArgumentParser
import torch
import numpy as np
from model import ESPCN
import onnx
import torch.npu
import os
NPU_CALCULATE_DEVICE = 0
if os.getenv('NPU_CALCULATE_DEVICE') and str.isdigit(os.getenv('NPU_CALCULATE_DEVICE')):
    NPU_CALCULATE_DEVICE = int(os.getenv('NPU_CALCULATE_DEVICE'))
if torch.npu.current_device() != NPU_CALCULATE_DEVICE:
    torch.npu.set_device(f'npu:{NPU_CALCULATE_DEVICE}')
log.basicConfig(level=log.INFO)


def convert_model(args):
    """ Function to convert the model from pytorch to specified format

    :param args: input args
    :return: Converted Model
    """

    model = ESPCN(num_channels=1, scaling_factor=args.scaling_factor)
    model.load_state_dict(torch.load(args.fpath_model))
    model.eval()
    model.to('cpu')

    # Sample Input Tensor
    x = torch.randn(1, 1, 224, 224, requires_grad=True)
    out = torch.randn(1, 1, x.shape[2] * args.scaling_factor, x.shape[3] * args.scaling_factor)
    print("input_shape: ", x.shape)
    print("output_shape: ", out.shape)

    if args.output_format == "ONNX":
        # Convert to ONNX format
        onnx_model_path = os.path.join(args.dirpath_out, f"espcn_model_x{args.scaling_factor}.onnx")
        torch.onnx.export(model, x, onnx_model_path,
                          export_params=True, opset_version=10, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        # Check ONNX Model
        onnx_model = onnx.load(os.path.join(args.dirpath_out, f"espcn_model_x{args.scaling_factor}.onnx"))
        onnx.checker.check_model(onnx_model)
        # Print a Human readable representation of the graph
        onnx.helper.printable_graph(onnx_model.graph)
        log.info(f"ONNX model saved at {os.path.join(args.dirpath_out, f'espcn_model_x{args.scaling_factor}.onnx')}\n")

    elif args.output_format == "CoreML":
        import coremltools as ct

        traced_model = torch.jit.trace(model, x)
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input", shape=x.shape)],
        )
        # Save model
        coreml_model.save(os.path.join(args.dirpath_out, f"espcn_model_x{args.scaling_factor}.mlmodel"))
        log.info(f"CoreML model saved at {os.path.join(args.dirpath_out, f'espcn_model_x{args.scaling_factor}.mlmodel')}\n")

    elif args.output_format == "TF" or args.output_format == "TFLite" or args.output_format == "TFjs":
        import tensorflow as tf
        from onnx_tf.backend import prepare

        # Convert to ONNX format
        onnx_model_path = os.path.join(args.dirpath_out, f"espcn_model_x{args.scaling_factor}.onnx")
        # Model output in SavedModel format
        tf_model_path = os.path.join(args.dirpath_out, f"espcn_saved_model_x{args.scaling_factor}")
        torch.onnx.export(model, x, os.path.join(args.dirpath_out, f"espcn_model_x{args.scaling_factor}.onnx"),
                          export_params=True, opset_version=10, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model)
        # Export TF Model
        tf_rep.export_graph(tf_model_path)
        # Test Converted Model
        model = tf.saved_model.load(tf_model_path)
        model.trainable = False
        input_tensor = tf.random.uniform([1, 1, 224, 224])
        out = model(**{'input': input_tensor})
        log.info(f"TensorFlow Model Output Shape: {out['output'].shape}")
        log.info(f"TensorFlow model saved at {tf_model_path}\n")

        # Convert TF to TFLite
        if args.output_format == "TFLite":
            # TFLite model path
            tflite_model_path = os.path.join(args.dirpath_out, f"espcn_saved_model_x{args.scaling_factor}.tflite")
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
                tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
            ]
            tflite_model = converter.convert()
            # Save the model
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model)

            # Test TFLite Model
            # Load the TFLite model and allocate tensors
            interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            # Get input and output tensors
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            # Test the model on random input data
            input_shape = input_details[0]['shape']
            input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            # get_tensor(): returns a copy of the tensor data
            output_data = interpreter.get_tensor(output_details[0]['index'])
            log.info(f"TFLite Output Shape: {output_data.shape}")
            log.info(f"TensorFlow Lite model saved at {tflite_model_path}")

        elif args.output_format == "TFjs":
            import subprocess

            tfjs_model_path = os.path.join(args.dirpath_out, f"espcn_tfjs_model_x{args.scaling_factor}")
            out = subprocess.run("tensorflowjs_converter "
                                 "--input_format=tf_saved_model "
                                 "--output_node_names='output' "
                                 "--saved_model_tags=serve "
                                 f"{tf_model_path} "
                                 f"{tfjs_model_path} ", shell=True)
            print(out)
            log.info(f"TensorFlow JS model saved at {tfjs_model_path}")

    else:
        print("Error. This format is not supported. Please check the list of available formats and try again")


def build_parser():
    parser = ArgumentParser(prog="ESPCN Export")
    parser.add_argument("-i", "--fpath_model", required=True, type=str,
                        help="Required. Path to saved model state dictionary.")
    parser.add_argument("-o", "--dirpath_out", required=True, type=str,
                        help="Required. Path to save exported model.")
    parser.add_argument("-sf", "--scaling_factor", default=3, required=False, type=int,
                        help="Optional. Image Up-scaling factor.")
    parser.add_argument("-f", "--output_format", required=True, type=str,
                        help="Required. Model export format. ex. ONNX, CoreML, TF, TFLite, TFjs")

    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    convert_model(args)