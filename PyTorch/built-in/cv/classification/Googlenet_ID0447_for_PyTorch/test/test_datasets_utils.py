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
#import os
import sys
import tempfile
import torchvision.datasets.utils as utils
import unittest
import zipfile
import tarfile
import gzip
import warnings
from torch._utils_internal import get_file_path_2

from common_utils import get_tmp_dir

if sys.version_info < (3,):
    from urllib2 import URLError
else:
    from urllib.error import URLError


TEST_FILE = get_file_path_2(
    os.path.dirname(os.path.abspath(__file__)), 'assets', 'grace_hopper_517x606.jpg')


with open('../url.ini', 'r') as f:
    content = f.read()
    vision_master_http_url = content.split('vision_master_http_url=')[1].split('\n')[0]
    vision_master__https_url = content.split('vision_master_http_url=')[1].split('\n')[0]
    vision_this_doesnt_exist_url = content.split('vision_master_http_url=')[1].split('\n')[0]


class Tester(unittest.TestCase):

    def test_check_md5(self):
        fpath = TEST_FILE
        correct_md5 = '9c0bb82894bb3af7f7675ef2b3b6dcdc'
        false_md5 = ''
        self.assertTrue(utils.check_md5(fpath, correct_md5))
        self.assertFalse(utils.check_md5(fpath, false_md5))

    def test_check_integrity(self):
        existing_fpath = TEST_FILE
        nonexisting_fpath = ''
        correct_md5 = '9c0bb82894bb3af7f7675ef2b3b6dcdc'
        false_md5 = ''
        self.assertTrue(utils.check_integrity(existing_fpath, correct_md5))
        self.assertFalse(utils.check_integrity(existing_fpath, false_md5))
        self.assertTrue(utils.check_integrity(existing_fpath))
        self.assertFalse(utils.check_integrity(nonexisting_fpath))

    def test_download_url(self):
        with get_tmp_dir() as temp_dir:
            url = vision_master_http_url
            try:
                utils.download_url(url, temp_dir)
                self.assertFalse(len(os.listdir(temp_dir)) == 0)
            except URLError:
                msg = "could not download test file '{}'".format(url)
                warnings.warn(msg, RuntimeWarning)
                raise unittest.SkipTest(msg)

    def test_download_url_retry_http(self):
        with get_tmp_dir() as temp_dir:
            url = vision_master__https_url
            try:
                utils.download_url(url, temp_dir)
                self.assertFalse(len(os.listdir(temp_dir)) == 0)
            except URLError:
                msg = "could not download test file '{}'".format(url)
                warnings.warn(msg, RuntimeWarning)
                raise unittest.SkipTest(msg)

    @unittest.skipIf(sys.version_info < (3,), "Python2 doesn't raise error")
    def test_download_url_dont_exist(self):
        with get_tmp_dir() as temp_dir:
            url = vision_this_doesnt_exist_url
            with self.assertRaises(URLError):
                utils.download_url(url, temp_dir)

    def test_extract_zip(self):
        with get_tmp_dir() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix='.zip') as f:
                with zipfile.ZipFile(f, 'w') as zf:
                    zf.writestr('file.tst', 'this is the content')
                utils.extract_archive(f.name, temp_dir)
                self.assertTrue(os.path.exists(os.path.join(temp_dir, 'file.tst')))
                with open(os.path.join(temp_dir, 'file.tst'), 'r') as nf:
                    data = nf.read()
                self.assertEqual(data, 'this is the content')

    def test_extract_tar(self):
        for ext, mode in zip(['.tar', '.tar.gz'], ['w', 'w:gz']):
            with get_tmp_dir() as temp_dir:
                with tempfile.NamedTemporaryFile() as bf:
                    bf.write("this is the content".encode())
                    bf.seek(0)
                    with tempfile.NamedTemporaryFile(suffix=ext) as f:
                        with tarfile.open(f.name, mode=mode) as zf:
                            zf.add(bf.name, arcname='file.tst')
                        utils.extract_archive(f.name, temp_dir)
                        self.assertTrue(os.path.exists(os.path.join(temp_dir, 'file.tst')))
                        with open(os.path.join(temp_dir, 'file.tst'), 'r') as nf:
                            data = nf.read()
                        self.assertEqual(data, 'this is the content')

    def test_extract_gzip(self):
        with get_tmp_dir() as temp_dir:
            with tempfile.NamedTemporaryFile(suffix='.gz') as f:
                with gzip.GzipFile(f.name, 'wb') as zf:
                    zf.write('this is the content'.encode())
                utils.extract_archive(f.name, temp_dir)
                f_name = os.path.join(temp_dir, os.path.splitext(os.path.basename(f.name))[0])
                self.assertTrue(os.path.exists(f_name))
                with open(os.path.join(f_name), 'r') as nf:
                    data = nf.read()
                self.assertEqual(data, 'this is the content')

    def test_verify_str_arg(self):
        self.assertEqual("a", utils.verify_str_arg("a", "arg", ("a",)))
        self.assertRaises(ValueError, utils.verify_str_arg, 0, ("a",), "arg")
        self.assertRaises(ValueError, utils.verify_str_arg, "b", ("a",), "arg")


if __name__ == '__main__':
    unittest.main()
