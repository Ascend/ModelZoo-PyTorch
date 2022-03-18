# Copyright 2020 Huawei Technologies Co., Ltd## Licensed under the Apache License, Version 2.0 (the "License");# you may not use this file except in compliance with the License.# You may obtain a copy of the License at## http://www.apache.org/licenses/LICENSE-2.0## Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.# See the License for the specific language governing permissions and# limitations under the License.# ============================================================================from .augmentations import (AudioAmplify, CenterCrop, ColorJitter, Flip, Fuse,
                            Imgaug, MelSpectrogram, MultiGroupCrop,
                            MultiScaleCrop, Normalize, RandomCrop,
                            RandomRescale, RandomResizedCrop, RandomScale,
                            Resize, TenCrop, ThreeCrop, TorchvisionTrans)
from .compose import Compose
from .formating import (Collect, FormatAudioShape, FormatShape, ImageToTensor,
                        Rename, ToDataContainer, ToTensor, Transpose)
from .loading import (AudioDecode, AudioDecodeInit, AudioFeatureSelector,
                      BuildPseudoClip, DecordDecode, DecordInit,
                      DenseSampleFrames, FrameSelector,
                      GenerateLocalizationLabels, ImageDecode,
                      LoadAudioFeature, LoadHVULabel, LoadLocalizationFeature,
                      LoadProposals, OpenCVDecode, OpenCVInit, PIMSDecode,
                      PIMSInit, PyAVDecode, PyAVDecodeMotionVector, PyAVInit,
                      RawFrameDecode, SampleAVAFrames, SampleFrames,
                      SampleProposalFrames, UntrimmedSampleFrames)
from .pose_loading import (GeneratePoseTarget, LoadKineticsPose, PoseDecode,
                           UniformSampleFrames)

__all__ = [
    'SampleFrames', 'PyAVDecode', 'DecordDecode', 'DenseSampleFrames',
    'OpenCVDecode', 'FrameSelector', 'MultiGroupCrop', 'MultiScaleCrop',
    'RandomResizedCrop', 'RandomCrop', 'Resize', 'Flip', 'Fuse', 'Normalize',
    'ThreeCrop', 'CenterCrop', 'TenCrop', 'ImageToTensor', 'Transpose',
    'Collect', 'FormatShape', 'Compose', 'ToTensor', 'ToDataContainer',
    'GenerateLocalizationLabels', 'LoadLocalizationFeature', 'LoadProposals',
    'DecordInit', 'OpenCVInit', 'PyAVInit', 'SampleProposalFrames',
    'UntrimmedSampleFrames', 'RawFrameDecode', 'DecordInit', 'OpenCVInit',
    'PyAVInit', 'SampleProposalFrames', 'ColorJitter', 'LoadHVULabel',
    'SampleAVAFrames', 'AudioAmplify', 'MelSpectrogram', 'AudioDecode',
    'FormatAudioShape', 'LoadAudioFeature', 'AudioFeatureSelector',
    'AudioDecodeInit', 'RandomScale', 'ImageDecode', 'BuildPseudoClip',
    'RandomRescale', 'PyAVDecodeMotionVector', 'Rename', 'Imgaug',
    'UniformSampleFrames', 'PoseDecode', 'LoadKineticsPose',
    'GeneratePoseTarget', 'PIMSInit', 'PIMSDecode', 'TorchvisionTrans'
]
