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
#

cdef extern from "src/region.h":
    ctypedef enum region_type "RegionType":
        EMTPY
        SPECIAL
        RECTANGEL
        POLYGON
        MASK

    ctypedef struct region_bounds:
        float top
        float bottom
        float left
        float right

    ctypedef struct region_rectangle:
        float x
        float y
        float width
        float height

    # ctypedef struct region_mask:
    #     int x
    #     int y
    #     int width
    #     int height
    #     char *data

    ctypedef struct region_polygon:
        int count
        float *x
        float *y

    ctypedef union region_container_data:
        region_rectangle rectangle
        region_polygon polygon
        # region_mask mask
        int special

    ctypedef struct region_container:
        region_type type
        region_container_data data

    # ctypedef struct region_overlap:
    #     float overlap
    #     float only1
    #     float only2

    # region_overlap region_compute_overlap(const region_container* ra, const region_container* rb, region_bounds bounds)

    float compute_polygon_overlap(const region_polygon* p1, const region_polygon* p2, float *only1, float *only2, region_bounds bounds)
