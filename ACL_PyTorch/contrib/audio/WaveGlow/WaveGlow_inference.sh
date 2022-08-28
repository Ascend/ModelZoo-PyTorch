#!/bin/bash

python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0001.wav.bin --dymDims mel:1,80,832 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0002.wav.bin --dymDims mel:1,80,164 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0003.wav.bin --dymDims mel:1,80,833 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0004.wav.bin --dymDims mel:1,80,443 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0005.wav.bin --dymDims mel:1,80,699 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0006.wav.bin --dymDims mel:1,80,490 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0007.wav.bin --dymDims mel:1,80,723 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0008.wav.bin --dymDims mel:1,80,154 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0009.wav.bin --dymDims mel:1,80,651 --output "./result" --outfmt BIN --batchsize 1
python3 ./tools/ais-bench_workload/tool/ais_infer/ais_infer.py --model "./waveglow_om.om" --input ./data/LJ001-0010.wav.bin --dymDims mel:1,80,760 --output "./result" --outfmt BIN --batchsize 1

