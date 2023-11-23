from llmtuner import run_exp
import torch_npu
from torch_npu.contrib import transfer_to_npu



def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()
	
def setup_seeds(seed=42):
    import random
    import numpy as np
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    cudnn.benchmark = False
    cudnn.deterministic = True



if __name__ == "__main__":
	import deepspeed
    import deepspeed_npu

    setup_seeds(42)
    torch.npu.set_compile_mode(jit_compile=False)
    deepspeed.init_distributed('hccl')

    main()
