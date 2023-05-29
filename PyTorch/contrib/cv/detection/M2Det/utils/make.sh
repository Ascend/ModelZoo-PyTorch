#import Cython

#cython _mask.pyx
#cython cpu_nms.pyx
python3 setup_cpu.py build_ext --inplace
