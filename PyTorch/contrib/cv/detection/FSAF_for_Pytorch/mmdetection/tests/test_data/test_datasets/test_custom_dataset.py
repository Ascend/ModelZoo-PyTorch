# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
#

from unittest.mock import MagicMock, patch

import pytest

from mmdet.datasets import DATASETS


@patch('mmdet.datasets.CocoDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.CustomDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.XMLDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.CityscapesDataset.load_annotations', MagicMock)
@patch('mmdet.datasets.CocoDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CustomDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.XMLDataset._filter_imgs', MagicMock)
@patch('mmdet.datasets.CityscapesDataset._filter_imgs', MagicMock)
@pytest.mark.parametrize('dataset',
                         ['CocoDataset', 'VOCDataset', 'CityscapesDataset'])
def test_custom_classes_override_default(dataset):
    dataset_class = DATASETS.get(dataset)
    if dataset in ['CocoDataset', 'CityscapesDataset']:
        dataset_class.coco = MagicMock()
        dataset_class.cat_ids = MagicMock()

    original_classes = dataset_class.CLASSES

    # Test setting classes as a tuple
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=('bus', 'car'),
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ('bus', 'car')

    # Test setting classes as a list
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['bus', 'car'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']

    # Test overriding not a subset
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=['foo'],
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['foo']

    # Test default behavior
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=None,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')

    assert custom_dataset.CLASSES == original_classes

    # Test sending file path
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile()
    with open(tmp_file.name, 'w') as f:
        f.write('bus\ncar\n')
    custom_dataset = dataset_class(
        ann_file=MagicMock(),
        pipeline=[],
        classes=tmp_file.name,
        test_mode=True,
        img_prefix='VOC2007' if dataset == 'VOCDataset' else '')
    tmp_file.close()

    assert custom_dataset.CLASSES != original_classes
    assert custom_dataset.CLASSES == ['bus', 'car']
