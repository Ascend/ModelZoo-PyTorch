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
import torch
from typing import Union, Any, List, Optional, MutableSequence

from pytorch_lightning.utilities import TPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def determine_root_gpu_device(gpus: List[int]) -> Optional[int]:
    """
    Args:
        gpus: non-empty list of ints representing which gpus to use

    Returns:
        designated root GPU device id
    """
    if gpus is None:
        return None

    assert isinstance(gpus, list), "gpus should be a list"
    assert len(gpus) > 0, "gpus should be a non empty list"

    # set root gpu
    root_gpu = gpus[0]

    return root_gpu


def parse_gpu_ids(gpus: Optional[Union[int, str, List[int]]]) -> Optional[List[int]]:
    """
    Parses the GPU ids given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        gpus: An int -1 or string '-1' indicate that all available GPUs should be used.
            A list of ints or a string containing list of comma separated integers
            indicates specific GPUs to use.
            An int 0 means that no GPUs should be used.
            Any int N > 0 indicates that GPUs [0..N) should be used.

    Returns:
        a list of gpus to be used or ``None`` if no GPUs were requested

    If no GPUs are available but the value of gpus variable indicates request for GPUs
    then a MisconfigurationException is raised.
    """

    # nothing was passed into the GPUs argument
    if callable(gpus):
        return None

    # Check that gpus param is None, Int, String or List
    _check_data_type(gpus)

    # Handle the case when no gpus are requested
    if gpus is None or isinstance(gpus, int) and gpus == 0:
        return None

    # We know user requested GPUs therefore if some of the
    # requested GPUs are not available an exception is thrown.

    gpus = _normalize_parse_gpu_string_input(gpus)
    gpus = _normalize_parse_gpu_input_to_list(gpus)
   # if not gpus:
   #	raise MisconfigurationException("GPUs requested but none are available.")
   # gpus = _sanitize_gpu_ids(gpus)

    return gpus


def parse_tpu_cores(tpu_cores: Union[int, str, List]) -> Optional[Union[List[int], int]]:
    """
    Parses the tpu_cores given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        tpu_cores: An int 1 or string '1' indicate that 1 core with multi-processing should be used
            An int 8 or string '8' indicate that all 8 cores with multi-processing should be used
            A list of int or a string containing list of comma separated integer
            indicates specific TPU core to use.

    Returns:
        a list of tpu_cores to be used or ``None`` if no TPU cores were requested
    """

    if callable(tpu_cores):
        return None

    _check_data_type(tpu_cores)

    if isinstance(tpu_cores, str):
        tpu_cores = _parse_tpu_cores_str(tpu_cores.strip())

    if not _tpu_cores_valid(tpu_cores):
        raise MisconfigurationException("`tpu_cores` can only be 1, 8 or [<1-8>]")

    if tpu_cores is not None and not TPU_AVAILABLE:
        raise MisconfigurationException('No TPU devices were found.')

    return tpu_cores


def _normalize_parse_gpu_string_input(s: Union[int, str, List[int]]) -> Union[int, List[int]]:
    if isinstance(s, str):
        if s == '-1':
            return -1
        else:
            return [int(x.strip()) for x in s.split(',') if len(x) > 0]
    else:
        return s


def _sanitize_gpu_ids(gpus: List[int]) -> List[int]:
    """
    Checks that each of the GPUs in the list is actually available.
    Raises a MisconfigurationException if any of the GPUs is not available.

    Args:
        gpus: list of ints corresponding to GPU indices

    Returns:
        unmodified gpus variable
    """
    all_available_gpus = _get_all_available_gpus()
    for gpu in gpus:
        if gpu not in all_available_gpus:
            raise MisconfigurationException(f"""
                You requested GPUs: {gpus}
                But your machine only has: {all_available_gpus}
            """)
    return gpus


def _normalize_parse_gpu_input_to_list(gpus: Union[int, List[int]]) -> Optional[List[int]]:
    assert gpus is not None
    if isinstance(gpus, MutableSequence):
        return list(gpus)

    # must be an int
    if not gpus:  # gpus==0
        return None
    if gpus == -1:
        return _get_all_available_gpus()

    return list(range(gpus))


def _get_all_available_gpus() -> List[int]:
    """
    Returns:
         a list of all available gpus
    """
    return list(range(torch.cuda.device_count()))


def _check_data_type(device_ids: Any) -> None:
    """
    Checks that the device_ids argument is one of: None, Int, String or List.
    Raises a MisconfigurationException otherwise.

    Args:
        device_ids: gpus/tpu_cores parameter as passed to the Trainer
    """
    if device_ids is not None and \
            (not isinstance(device_ids, (int, str, MutableSequence)) or isinstance(device_ids, bool)):
        raise MisconfigurationException("Device ID's (GPU/TPU) must be int, string or sequence of ints or None.")


def _tpu_cores_valid(tpu_cores):
    # allow 1 or 8 cores
    if tpu_cores in (1, 8, None):
        return True

    # allow picking 1 of 8 indexes
    if isinstance(tpu_cores, (list, tuple, set)):
        has_1_tpu_idx = len(tpu_cores) == 1
        is_valid_tpu_idx = tpu_cores[0] in range(1, 9)

        is_valid_tpu_core_choice = has_1_tpu_idx and is_valid_tpu_idx
        return is_valid_tpu_core_choice

    return False


def _parse_tpu_cores_str(tpu_cores):
    if tpu_cores in ('1', '8'):
        tpu_cores = int(tpu_cores)
    else:
        tpu_cores = [int(x.strip()) for x in tpu_cores.split(',') if len(x) > 0]
    return tpu_cores
