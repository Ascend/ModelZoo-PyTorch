#     Copyright 2021 Huawei Technologies Co., Ltd
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

"""Superpowers which should be rarely used.

This library contains functions for importing python classes and
for running shell commands. Remember, with great power comes great
responsibility.

Authors
 * Mirco Ravanelli 2020
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def run_shell(cmd):
    r"""This function can be used to run a command in the bash shell.

    Arguments
    ---------
    cmd : str
        Shell command to run.

    Returns
    -------
    bytes
        The captured standard output.
    bytes
        The captured standard error.
    int
        The returncode.

    Raises
    ------
    OSError
        If returncode is not 0, i.e., command failed.

    Example
    -------
    >>> out, err, code = run_shell("echo 'hello world'")
    >>> out.decode("utf-8")
    'hello world\n'
    """

    # Executing the command
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    # Capturing standard output and error
    (output, err) = p.communicate()

    if p.returncode != 0:
        raise OSError(err.decode("utf-8"))

    # Adding information in the logger
    msg = output.decode("utf-8") + "\n" + err.decode("utf-8")
    logger.debug(msg)

    return output, err, p.returncode
