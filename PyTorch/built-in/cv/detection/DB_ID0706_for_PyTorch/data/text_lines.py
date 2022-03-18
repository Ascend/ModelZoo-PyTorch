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

from .quad import Quad


class TextLines:
    '''
    The abstrct class of text lines in an input image for scene text detection and recognition.
    Input:
        lines:
            - text: the text content of a text instance.
              poly: the quadrangle-box of the text instance.
              charboxes: the quadrangle-box of the characters inside the corresponding text instance.
    '''
    def __init__(self, lines, with_charboxes=True):
        self.texts = []
        quads = []
        self.charboxes = []
        for line in lines:
            self.texts.append(line['text'])
            quads.append(line['poly'])
            if with_charboxes and 'charboxes' in line:
                self.charboxes.append(Quad(line['charboxes']))
        self.with_charboxes = len(self.charboxes) > 0
        self.quads = Quad(quads)
        self._rects = None

    def __iter__(self):
        for text, quad in zip(self.texts, self.quads):
            yield(text, quad)

    @property
    def rects(self):
        if self._rects is None:
            self._rects = self.quads.rectify()
        return self._rects

    def __len__(self):
        return len(self.texts)

    def char_count(self):
        return sum([len(t) for t in self.texts])
