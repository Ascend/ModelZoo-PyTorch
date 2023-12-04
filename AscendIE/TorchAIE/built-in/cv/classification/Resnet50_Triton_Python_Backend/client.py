import argparse
import json
import sys
import warnings

import numpy as np
import torch
import tritonclient.http as httpclient
from tritonclient.utils import *

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="resnet50",
        help="Model name",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=False,
        default="localhost:8000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        triton_client = httpclient.InferenceServerClient(args.url)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if args.verbose:
        print(json.dumps(triton_client.get_model_config(args.model_name), indent=4))

    input_name = "INPUT"
    output_name = "OUTPUT"

    shape = [4, 3, 224, 224]
    input_data = np.random.rand(*shape).astype(np.float32)
    inputs = [
        httpclient.InferInput(
            "INPUT", input_data.shape, np_to_triton_dtype(input_data.dtype)
        )
    ]
    output = httpclient.InferRequestedOutput(output_name)

    inputs[0].set_data_from_numpy(input_data)
    results = triton_client.infer(model_name=args.model_name, inputs=inputs, outputs=[output])

    output_data = results.as_numpy(output_name)

    print(
        "INPUT ({}) , OUTPUT ({})".format(
            input_data, output_data
        )
    )

    print("PASS: ResNet50 instance kind")
    sys.exit(0)
