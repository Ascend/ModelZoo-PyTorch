# -*- coding: utf-8 -*-
# BSD 3-Clause License
#
# Copyright (c) 2017
# All rights reserved.
# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==========================================================================

# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import mmcv

try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging_utils import ModelExportUtils
except ImportError:
    package_model = None


def mmocr2torchserve(
    config_file: str,
    checkpoint_file: str,
    output_folder: str,
    model_name: str,
    model_version: str = '1.0',
    force: bool = False,
):
    """Converts MMOCR model (config + checkpoint) to TorchServe `.mar`.

    Args:
        config_file:
            In MMOCR config format.
            The contents vary for each task repository.
        checkpoint_file:
            In MMOCR checkpoint format.
            The contents vary for each task repository.
        output_folder:
            Folder where `{model_name}.mar` will be created.
            The file created will be in TorchServe archive format.
        model_name:
            If not None, used for naming the `{model_name}.mar` file
            that will be created under `output_folder`.
            If None, `{Path(checkpoint_file).stem}` will be used.
        model_version:
            Model's version.
        force:
            If True, if there is an existing `{model_name}.mar`
            file under `output_folder` it will be overwritten.
    """
    mmcv.mkdir_or_exist(output_folder)

    config = mmcv.Config.fromfile(config_file)

    with TemporaryDirectory() as tmpdir:
        config.dump(f'{tmpdir}/config.py')

        args = Namespace(
            **{
                'model_file': f'{tmpdir}/config.py',
                'serialized_file': checkpoint_file,
                'handler': f'{Path(__file__).parent}/mmocr_handler.py',
                'model_name': model_name or Path(checkpoint_file).stem,
                'version': model_version,
                'export_path': output_folder,
                'force': force,
                'requirements_file': None,
                'extra_files': None,
                'runtime': 'python',
                'archive_format': 'default'
            })
        manifest = ModelExportUtils.generate_manifest_json(args)
        package_model(args, manifest)


def parse_args():
    parser = ArgumentParser(
        description='Convert MMOCR models to TorchServe `.mar` format.')
    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('checkpoint', type=str, help='checkpoint file path')
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Folder where `{model_name}.mar` will be created.')
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        help='If not None, used for naming the `{model_name}.mar`'
        'file that will be created under `output_folder`.'
        'If None, `{Path(checkpoint_file).stem}` will be used.')
    parser.add_argument(
        '--model-version',
        type=str,
        default='1.0',
        help='Number used for versioning.')
    parser.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='overwrite the existing `{model_name}.mar`')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if package_model is None:
        raise ImportError('`torch-model-archiver` is required.'
                          'Try: pip install torch-model-archiver')

    mmocr2torchserve(args.config, args.checkpoint, args.output_folder,
                     args.model_name, args.model_version, args.force)
