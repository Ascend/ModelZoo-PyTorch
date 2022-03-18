# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from distutils.core import setup
from distutils.extension import Extension
from os.path import join as pjoin

import numpy as np
from Cython.Distutils import build_ext


def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path('nvcc',
                            os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError(
                'The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME'
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        'home': home,
        'nvcc': nvcc,
        'include': pjoin(home, 'include'),
        'lib64': pjoin(home, 'lib64')
    }
    for k, v in cudaconfig.items():
        # for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            raise EnvironmentError(
                'The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig


# CUDA = locate_cuda()

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(extra_postargs)
        """
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']
        """
        postargs = extra_postargs['gcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


"""
ext_modules = [
    Extension(
        "utils.bbox",
        ["bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension(
        "utils.cython_nms",
        ["cython_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
               
    Extension('utils.gpu_nms',
        ['nms_kernel.cu', 'gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_35',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]},
        include_dirs = [numpy_include, CUDA['include']]
    ),
]
"""
ext_modules = [
    Extension(
        "nms.cpu_nms",
        ["nms/cpu_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs=[numpy_include]
    ),
    Extension(
        'pycocotools._mask',
        sources=['pycocotools/maskApi.c', 'pycocotools/_mask.pyx'],
        include_dirs=[numpy_include, 'pycocotools'],
        extra_compile_args={
            'gcc': ['-Wno-cpp', '-Wno-unused-function', '-std=c99']},
    ),
]

setup(
    name='mot_utils',
    ext_modules=ext_modules,
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},
)
