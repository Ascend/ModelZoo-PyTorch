# Copyright The PyTorch Lightning team.
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

import ast
import csv
import inspect
import os
from argparse import Namespace
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, IO, MutableMapping, Optional, Union
from warnings import warn

import torch
import yaml

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import AttributeDict, OMEGACONF_AVAILABLE, rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.parsing import parse_class_init_keys

PRIMITIVE_TYPES = (bool, int, float, str)
ALLOWED_CONFIG_TYPES = (AttributeDict, MutableMapping, Namespace)

if OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig
    from omegaconf.errors import UnsupportedValueType, ValidationError


# the older shall be on the top
CHECKPOINT_PAST_HPARAMS_KEYS = (
    'hparams',
    'module_arguments',  # used in 0.7.6
)


class ModelIO(object):
    CHECKPOINT_HYPER_PARAMS_KEY = 'hyper_parameters'
    CHECKPOINT_HYPER_PARAMS_NAME = 'hparams_name'
    CHECKPOINT_HYPER_PARAMS_TYPE = 'hparams_type'

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        **kwargs,
    ):
        r"""
        Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint
        it stores the arguments passed to `__init__`  in the checkpoint under `hyper_parameters`

        Any arguments specified through \*args and \*\*kwargs will override args stored in `hyper_parameters`.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a .yaml file with hierarchical structure
                as in this example::

                    drop_prob: 0.2
                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a .yaml file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningModule` for use.

                If your model's `hparams` argument is :class:`~argparse.Namespace`
                and .yaml file has hierarchical structure, you need to refactor your model to treat
                `hparams` as :class:`~dict`.
            strict: Whether to strictly enforce that the keys in :attr:`checkpoint_path` match the keys
                returned by this module's state dict. Default: `True`.
            kwargs: Any extra keyword args needed to init the model. Can also be used to override saved
                hyperparameter values.

        Return:
            :class:`LightningModule` with loaded weights and hyperparameters (if available).

        Example:
            .. code-block:: python

                # load weights without mapping ...
                MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

                # or load weights mapping all weights from GPU 1 to GPU 0 ...
                map_location = {'cuda:1':'cuda:0'}
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    map_location=map_location
                )

                # or load weights and hyperparameters from separate files.
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    hparams_file='/path/to/hparams_file.yaml'
                )

                # override some of the params with new values
                MyLightningModule.load_from_checkpoint(
                    PATH,
                    num_layers=128,
                    pretrained_ckpt_path: NEW_PATH,
                )

                # predict
                pretrained_model.eval()
                pretrained_model.freeze()
                y_hat = pretrained_model(x)
        """
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        if hparams_file is not None:
            extension = hparams_file.split('.')[-1]
            if extension.lower() in ('csv'):
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ('yml', 'yaml'):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError('.csv, .yml or .yaml is required for `hparams_file`')

            hparams['on_gpu'] = False

            # overwrite hparams by the given file
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

        # for past checkpoint need to add the new key
        if cls.CHECKPOINT_HYPER_PARAMS_KEY not in checkpoint:
            checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = {}
        # override the hparams with values that were passed in
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

        model = cls._load_model_state(checkpoint, strict=strict, **kwargs)
        return model

    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], strict: bool = True, **cls_kwargs_new):
        cls_spec = inspect.getfullargspec(cls.__init__)
        cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

        self_var, args_var, kwargs_var = parse_class_init_keys(cls)
        drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
        cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

        cls_kwargs_loaded = {}
        # pass in the values we saved automatically
        if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:

            # 1. (backward compatibility) Try to restore model hparams from checkpoint using old/past keys
            for _old_hparam_key in CHECKPOINT_PAST_HPARAMS_KEYS:
                cls_kwargs_loaded.update(checkpoint.get(_old_hparam_key, {}))

            # 2. Try to restore model hparams from checkpoint using the new key
            _new_hparam_key = cls.CHECKPOINT_HYPER_PARAMS_KEY
            cls_kwargs_loaded.update(checkpoint.get(_new_hparam_key))

            # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
            cls_kwargs_loaded = _convert_loaded_hparams(cls_kwargs_loaded, checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_TYPE))

            # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
            args_name = checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_NAME)
            if args_name and args_name in cls_init_args_name:
                cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

        _cls_kwargs = {}
        _cls_kwargs.update(cls_kwargs_loaded)
        _cls_kwargs.update(cls_kwargs_new)

        if not cls_spec.varkw:
            # filter kwargs according to class init unless it allows any argument via kwargs
            _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

        model = cls(**_cls_kwargs)

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        # load the state_dict on the model automatically
        model.load_state_dict(checkpoint['state_dict'], strict=strict)

        return model

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Do something with the checkpoint.
        Gives model a chance to load something before ``state_dict`` is restored.

        Args:
            checkpoint: A dictionary with variables from the checkpoint.
        """

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Give the model a chance to add something to the checkpoint.
        ``state_dict`` is already there.

        Args:
            checkpoint: A dictionary in which you can save variables to save in a checkpoint.
                Contents need to be pickleable.
        """

    # -------------------------
    # OPTIONAL HOOKS
    # -------------------------
    def on_hpc_save(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook to do whatever you need right before Slurm manager saves the model.

        Args:
            checkpoint: A dictionary in which you can save variables to save in a checkpoint.
                Contents need to be pickleable.
        """

    def on_hpc_load(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook to do whatever you need right before Slurm manager loads the model.

        Args:
            checkpoint: A dictionary with variables from the checkpoint.
        """


def _convert_loaded_hparams(model_args: dict, hparams_type: Optional[Union[Callable, str]] = None) -> object:
    """Convert hparams according given type in callable or string (past) format."""
    # if not hparams type define
    if not hparams_type:
        return model_args
    # if past checkpoint loaded, convert str to callable
    if isinstance(hparams_type, str):
        hparams_type = AttributeDict
    # convert hparams
    return hparams_type(model_args)


def update_hparams(hparams: dict, updates: dict) -> None:
    """
    Overrides hparams with new values

    >>> hparams = {'c': 4}
    >>> update_hparams(hparams, {'a': {'b': 2}, 'c': 1})
    >>> hparams['a']['b'], hparams['c']
    (2, 1)
    >>> update_hparams(hparams, {'a': {'b': 4}, 'c': 7})
    >>> hparams['a']['b'], hparams['c']
    (4, 7)

    Args:
        hparams: the original params and also target object
        updates: new params to be used as update

    """
    for k, v in updates.items():
        # if missing, add the key
        if k not in hparams:
            hparams[k] = v
            continue

        # recurse if dictionary
        if isinstance(v, dict):
            update_hparams(hparams[k], updates[k])
        else:
            # update the value
            hparams.update({k: v})


def load_hparams_from_tags_csv(tags_csv: str) -> Dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_csv = os.path.join('.', 'testing-hparams.csv')
    >>> save_hparams_to_tags_csv(path_csv, hparams)
    >>> hparams_new = load_hparams_from_tags_csv(path_csv)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_csv)
    """
    fs = get_filesystem(tags_csv)
    if not fs.exists(tags_csv):
        rank_zero_warn(f"Missing Tags: {tags_csv}.", RuntimeWarning)
        return {}

    with fs.open(tags_csv, "r", newline="") as fp:
        csv_reader = csv.reader(fp, delimiter=",")
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}

    return tags


def save_hparams_to_tags_csv(tags_csv: str, hparams: Union[dict, Namespace]) -> None:
    fs = get_filesystem(tags_csv)
    if not fs.isdir(os.path.dirname(tags_csv)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(tags_csv)}.")

    if isinstance(hparams, Namespace):
        hparams = vars(hparams)

    with fs.open(tags_csv, "w", newline="") as fp:
        fieldnames = ["key", "value"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writerow({"key": "key", "value": "value"})
        for k, v in hparams.items():
            writer.writerow({"key": k, "value": v})


def load_hparams_from_yaml(config_yaml: str, use_omegaconf: bool = True) -> Dict[str, Any]:
    """Load hparams from a file.

        Args:
            config_yaml: Path to config yaml file
            use_omegaconf: If both `OMEGACONF_AVAILABLE` and `use_omegaconf` are True,
                the hparams will be converted to `DictConfig` if possible

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)
    """
    fs = get_filesystem(config_yaml)
    if not fs.exists(config_yaml):
        rank_zero_warn(f"Missing Tags: {config_yaml}.", RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        hparams = yaml.full_load(fp)

    if OMEGACONF_AVAILABLE:
        if use_omegaconf:
            try:
                return OmegaConf.create(hparams)
            except (UnsupportedValueType, ValidationError):
                pass
    return hparams


def save_hparams_to_yaml(config_yaml, hparams: Union[dict, Namespace]) -> None:
    """
    Args:
        config_yaml: path to new YAML file
        hparams: parameters to be saved
    """
    fs = get_filesystem(config_yaml)
    if not fs.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace or AD to dict
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)
    elif isinstance(hparams, AttributeDict):
        hparams = dict(hparams)

    # saving with OmegaConf objects
    if OMEGACONF_AVAILABLE:
        # deepcopy: hparams from user shouldn't be resolved
        hparams = deepcopy(hparams)
        to_container = partial(OmegaConf.to_container, resolve=True)
        hparams = apply_to_collection(hparams, DictConfig, to_container)
        with fs.open(config_yaml, "w", encoding="utf-8") as fp:
            try:
                OmegaConf.save(hparams, fp)
                return
            except (UnsupportedValueType, ValidationError):
                pass

    assert isinstance(hparams, dict)
    hparams_allowed = {}
    # drop paramaters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        try:
            yaml.dump(v)
        except TypeError as err:
            warn(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
            hparams[k] = type(v).__name__
        else:
            hparams_allowed[k] = v

    # saving the standard way
    with fs.open(config_yaml, "w", newline="") as fp:
        yaml.dump(hparams_allowed, fp)


def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as err:
        log.debug(err)
        return val
