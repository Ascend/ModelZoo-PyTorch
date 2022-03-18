# coding=utf-8

#     Copyright [yyyy] [name of copyright owner]
#     Copyright 2020 Huawei Technologies Co., Ltd
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#



import logging


def add_to_path(path):
    '''
    add path to sys.path.
    '''
    import sys;
    sys.path.insert(0, path);


def add_ancester_dir_to_path(fp, p):
    '''
    add ancester directory to sys.path.
    fp: usually __file__
    p : the relative path to be added.
    '''
    import util
    parent_path = util.io.get_dir(fp)
    path = util.io.join_path(parent_path, p)
    add_to_path(path)


def is_main(mod_name):
    return mod_name == '__main__'


def import_by_name(mod_name):
    __import__(mod_name)
    return get_mod_by_name(mod_name)


def try_import_by_name(mod_name, error_path):
    try:
        import_by_name(mod_name)
    except ImportError:
        logging.info('adding %s to sys.path' % (error_path))
        add_to_path(error_path)
        import_by_name(mod_name)

    return get_mod_by_name(mod_name)


def get_mod_by_name(mod_name):
    import sys
    return sys.modules[mod_name]


def load_mod_from_path(path, keep_name=True):
    """"
    Params:
        path
        keep_name: if True, the filename will be used as module name.
    """
    import util
    import imp
    path = util.io.get_absolute_path(path)
    file_name = util.io.get_filename(path)
    module_name = file_name.split('.')[0]
    if not keep_name:
        module_name = '%s_%d' % (module_name, util.get_count())
    return imp.load_source(module_name, path)
