#
# BSD 3-Clause License
#
# Copyright (c) 2017 xxxx
# All rights reserved.
# Copyright 2021 Huawei Technologies Co., Ltd
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
# ============================================================================
#from PIL import Image
from six.moves import zip
from .utils import download_url, check_integrity

import os
from .vision import VisionDataset


class SBU(VisionDataset):
    """`SBU Captioned Photo <http://www.cs.virginia.edu/~vicente/sbucaptions/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where tarball
            ``SBUCaptionedPhotoDataset.tar.gz`` exists.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    with open('../../url.ini', 'r') as _f:
        _content = _f.read()
        url = _content.split('sbucaption_dataset_url=')[1].split('\n')[0]
    filename = "SBUCaptionedPhotoDataset.tar.gz"
    md5_checksum = '9aec147b3488753cf758b4d493422285'

    def __init__(self, root, transform=None, target_transform=None, download=True):
        super(SBU, self).__init__(root, transform=transform,
                                  target_transform=target_transform)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # Read the caption for each photo
        self.photos = []
        self.captions = []

        file1 = os.path.join(self.root, 'dataset', 'SBU_captioned_photo_dataset_urls.txt')
        file2 = os.path.join(self.root, 'dataset', 'SBU_captioned_photo_dataset_captions.txt')

        for line1, line2 in zip(open(file1), open(file2)):
            url = line1.rstrip()
            photo = os.path.basename(url)
            filename = os.path.join(self.root, 'dataset', photo)
            if os.path.exists(filename):
                caption = line2.rstrip()
                self.photos.append(photo)
                self.captions.append(caption)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a caption for the photo.
        """
        filename = os.path.join(self.root, 'dataset', self.photos[index])
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        target = self.captions[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """The number of photos in the dataset."""
        return len(self.photos)

    def _check_integrity(self):
        """Check the md5 checksum of the downloaded tarball."""
        root = self.root
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.md5_checksum):
            return False
        return True

    def download(self):
        """Download and extract the tarball, and download each individual photo."""
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.md5_checksum)

        # Extract file
        with tarfile.open(os.path.join(self.root, self.filename), 'r:gz') as tar:
            tar.extractall(path=self.root)

        # Download individual photos
        with open(os.path.join(self.root, 'dataset', 'SBU_captioned_photo_dataset_urls.txt')) as fh:
            for line in fh:
                url = line.rstrip()
                try:
                    download_url(url, os.path.join(self.root, 'dataset'))
                except OSError:
                    # The images point to public images on Flickr.
                    # Note: Images might be removed by users at anytime.
                    pass
