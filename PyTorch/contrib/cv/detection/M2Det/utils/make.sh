#import Cython

#cython _mask.pyx
#cython cpu_nms.pyx
python3.7 setup_cpu.py build_ext --inplace
