import argparse
import asyncio
import json
import logging
import os
import sys
import time

import aclruntime
from tqdm import tqdm

from frontend.io_oprations import (create_infileslist_from_inputs_list,
                          create_intensors_from_infileslist,
                          create_intensors_zerodata,
                          get_tensor_from_files_list, pure_infer_dump_file,
                          save_tensors_to_file)
from frontend.summary import summary
from frontend.utils import logger


def set_session_options(session, args):
    # 增加校验
    if args.dymBatch != 0:
        session.set_dynamic_batchsize(args.dymBatch)
    elif args.dymHW !=None:
        hwstr = args.dymHW.split(",")
        session.set_dynamic_hw((int)(hwstr[0]), (int)(hwstr[1]))
    elif args.dymDims !=None:
        session.set_dynamic_dims(args.dymDims)
    elif args.dymShape !=None:
        session.set_dynamic_shape(args.dymShape)
    else:
        session.set_staticbatch()

    # 设置custom out tensors size
    if args.outputSize != None:
        customsizes = [int(n) for n in args.outputSize.split(',')]
        logger.debug("set customsize:{}".format(customsizes))
        session.set_custom_outsize(customsizes)

def init_inference_session(args):
    options = aclruntime.session_options()
    if args.acl_json_path != None:
        options.acl_json_path = args.acl_json_path
    if args.debug == True:
        logger.setLevel(logging.DEBUG)
        options.log_level = 1
    session = aclruntime.InferenceSession(args.model, args.device_id, options)

    set_session_options(session, args)
    logger.debug("session info:{}".format(session))
    return session

def run_inference(session, inputs, outputs_names, loop=1):
    session.run_setinputs(inputs)
    starttime = time.time()
    session.run_execute(loop)
    endtime = time.time()
    summary.npu_compute_time_list.append(float(endtime - starttime) * 1000.0/loop)  # millisecond
    outputs = session.run_getoutputs(outputs_names)

    return outputs

def warmup(session, args, intensors_desc, outputs_names):
    n_loop = 5
    inputs = create_intensors_zerodata(intensors_desc, args.device_id, args.pure_data_type)
    for i in range(n_loop):
        run_inference(session, inputs, outputs_names, 1)
    summary.reset()
    session.reset_sumaryinfo()
    logger.debug("warm up {} times done".format(n_loop))

# 轮训运行推理
def infer_loop_run(session, args, intensors_desc, infileslist, outputs_names, output_prefix):
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference Processing')):
        intensors = []
        for j, files in enumerate(infiles):
            tensor = get_tensor_from_files_list(files, args.device_id, intensors_desc[j].realsize, args.pure_data_type)
            intensors.append(tensor)
        outputs = run_inference(session, intensors, outputs_names, args.loop)
        if args.output != None:
            save_tensors_to_file(outputs, output_prefix, infiles, args.outfmt, i)

# 先准备好数据 然后执行推理 然后统一写文件
def infer_fulltensors_run(session, args, intensors_desc, infileslist, outputs_names, output_prefix):
    outtensors = []
    intensorslist = create_intensors_from_infileslist(infileslist, intensors_desc, args.device_id, args.pure_data_type)

    #for inputs in intensorslist:
    for inputs in tqdm(intensorslist, file=sys.stdout, desc='Inference Processing full'):
        outputs = run_inference(session, inputs, outputs_names, args.loop)
        outtensors.append(outputs)

    if args.output != None:
        for i, outputs in enumerate(outtensors):
            save_tensors_to_file(outputs, output_prefix, infileslist[i], args.outfmt, i)

async def in_task(inque, args, intensors_desc, infileslist):
    logger.debug("in_task begin")
    for i, infiles in enumerate(tqdm(infileslist, file=sys.stdout, desc='Inference Processing task')):
        intensors = []
        for j, files in enumerate(infiles):
            tensor = get_tensor_from_files_list(files, args.device_id, intensors_desc[j].realsize, args.pure_data_type)
            intensors.append(tensor)
        await inque.put([intensors, infiles, i])
    await inque.put([None, None, None])
    logger.debug("in_task exit")

async def infer_task(inque, session, outputs_names, args, outque):
    logger.debug("infer_task begin")
    while True:
        intensors, infiles, i = await inque.get()
        if intensors == None:
            await outque.put([None, None, None])
            logger.debug("infer_task exit")
            break
        outputs = run_inference(session, intensors, outputs_names, args.loop)
        await outque.put([outputs, infiles, i])

async def out_task(outque, output_prefix, args):
    logger.debug("out_task begin")
    while True:
        outputs, infiles, i = await outque.get()
        if outputs == None:
            logger.debug("out_task exit")
            break
        if args.output != None:
            save_tensors_to_file(outputs, output_prefix, infiles, args.outfmt, i)

async def infer_pipeline_process_run(session, args, intensors_desc, infileslist, outputs_names, output_prefix):
    inque = asyncio.Queue(maxsize=20)
    outque = asyncio.Queue(maxsize=20)

    await asyncio.gather(
        in_task(inque, args, intensors_desc, infileslist),
        infer_task(inque, session, outputs_names, args, outque),
        out_task(outque, output_prefix, args),
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--om", required=True, help="the path of the om model")
    parser.add_argument("--input", "-i", default=None, help="input file or dir")
    parser.add_argument("--output", "-o", default=None, help="output")
    parser.add_argument("--output_prefix", default=None, help="output_prefix")
    parser.add_argument("--outfmt", default="BIN", choices=["NPY", "BIN"], help="Output file format (NPY or BIN)")
    parser.add_argument("--loop", "-r", type=int, default=1, help="the round of the PrueInfer.")
    parser.add_argument("--debug", action="store_true", help="Debug switch,print model information")
    parser.add_argument("--device_id", "--device", type=int, default=0, choices=range(0, 255), help="the NPU device ID to use")
    parser.add_argument("--dymBatch", type=int, default=0, help="dynamic batch size param，such as --dymBatch 2")
    parser.add_argument("--dymHW", type=str, default=None, help="dynamic image size param, such as --dymHW \"300,500\"")
    parser.add_argument("--dymDims", type=str, default=None, help="dynamic dims param, such as --dymDims \"data:1,600;img_info:1,600\"")
    parser.add_argument("--dymShape", type=str, help="dynamic hape param, such as --dymShape \"data:1,600;img_info:1,600\"")
    parser.add_argument("--outputSize", type=str, default=None, help="output size for dynamic shape mode")
    parser.add_argument("--acl_json_path", type=str, default=None, help="acl json path for profiling or dump")
    parser.add_argument("--batchsize", type=int, default=1, help="batch size of input tensor")
    parser.add_argument("--pure_data_type", type=str, default="zero", choices=["zero", "random"], help="null data type for pure inference(zero or random")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    session = init_inference_session(args)

    intensors_desc = session.get_inputs()
    outtensors_desc = session.get_outputs()

    if args.output != None:
        timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
        if args.output_prefix is not None:
            output_prefix = os.path.join(args.output, args.output_prefix)
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix, 0o755)
        else:
            output_prefix = os.path.join(args.output, timestr)
            os.mkdir(output_prefix, 0o755)
        logger.info("output path:{}".format(output_prefix))
    else:
        output_prefix = None

    outputs_names = [desc.name for desc in outtensors_desc ]

    warmup(session, args, intensors_desc, outputs_names)

    inputs_list = [] if args.input == None else args.input.split(',')

    # create infiles list accord inputs list
    if len(inputs_list) == 0:
        # 纯推理场景 创建输入zero数据
        infileslist = [[ [ pure_infer_dump_file ] for index in intensors_desc ]]
    else:
        infileslist = create_infileslist_from_inputs_list(inputs_list, intensors_desc)

    #infer_fulltensors_run(session, args, intensors_desc, infileslist, outputs_names, output_prefix)
    #infer_loop_run(session, args, intensors_desc, infileslist, outputs_names, output_prefix)
    asyncio.run(infer_pipeline_process_run(session, args, intensors_desc, infileslist, outputs_names, output_prefix))

    summary.add_args(sys.argv)
    summary.report(args.batchsize, output_prefix)

    #print(session.sumary())
