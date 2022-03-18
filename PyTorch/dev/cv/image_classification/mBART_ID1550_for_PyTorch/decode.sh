#!/bin/bash

python decode.py -data data-bin/demo \
	-save_model demo-model \
	--valid_batch_size 1 \
	--gpu_ranks 0 \
#	--tgt_lang_id DE \
