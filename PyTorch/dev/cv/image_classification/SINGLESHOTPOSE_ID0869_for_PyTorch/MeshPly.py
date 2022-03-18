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
# Class to read 

class MeshPly:
    def __init__(self, filename, color=[0., 0., 0.]):

        f = open(filename, 'r')
        self.vertices = []
        self.colors = []
        self.indices = []
        self.normals = []

        vertex_mode = False
        face_mode = False

        nb_vertices = 0
        nb_faces = 0

        idx = 0

        with f as open_file_object:
            for line in open_file_object:
                elements = line.split()
                if vertex_mode:
                    self.vertices.append([float(i) for i in elements[:3]])
                    self.normals.append([float(i) for i in elements[3:6]])

                    if elements[6:9]:
                        self.colors.append([float(i) / 255. for i in elements[6:9]])
                    else:
                        self.colors.append([float(i) / 255. for i in color])

                    idx += 1
                    if idx == nb_vertices:
                        vertex_mode = False
                        face_mode = True
                        idx = 0
                elif face_mode:
                    self.indices.append([float(i) for i in elements[1:4]])
                    idx += 1
                    if idx == nb_faces:
                        face_mode = False
                elif elements[0] == 'element':
                    if elements[1] == 'vertex':
                        nb_vertices = int(elements[2])
                    elif elements[1] == 'face':
                        nb_faces = int(elements[2])
                elif elements[0] == 'end_header':
                    vertex_mode = True


if __name__ == '__main__':
    path_model = ''
    mesh = MeshPly(path_model)
