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
import argparse
import json
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *

from vgg_pytorch import VGG

parser = argparse.ArgumentParser("VGGNet Classifier Tool")
parser.add_argument("-w", "--model_name", type=str, default='vgg11',
                    help="Weight of the model loaded by default.")
parser.add_argument("-s", "--image_size", type=int, default=None,
                    help="Size of classified image. (default=None).")
parser.add_argument("-l", "--labels_map", type=str, default="./labels_map.txt",
                    help="Image tag. (default='./labels_map.txt').")
parser.add_argument("-n", "--num_classes", type=int, default=1000,
                    help="Number of categories of images. (default=1000).")
parser.add_argument("-p", "--echo", type=bool, default=False,
                    help="Show pop ups or not? (default: False)")
args = parser.parse_args()


def classifier(image_path):
  # Open image
  img = Image.open(image_path)
  img = tfms(img).unsqueeze(0)

  # Classify with VGGNet
  with torch.no_grad():
    logits = model(img)
  preds = torch.topk(logits, k=1).indices.squeeze(0).tolist()

  for idx in preds:
    label = labels_map[idx]
    probability = torch.softmax(logits, dim=1)[0, idx].item()
  return label, probability


class Picture(QWidget):
  def __init__(self):
    super(Picture, self).__init__()

    self.resize(1000, 1000)
    self.setWindowTitle("Classifier tool")

    self.label = QLabel(self)
    self.label.setFixedSize(args.image_size, args.image_size)
    self.label.move(300, 300)
    self.label.setStyleSheet(
      "QLabel{background:white;}"
      "QLabel{color:rgb(0,0,0);font-size:18px;font-weight:bold;font-family:宋体;}"
    )

    # add open image button
    self.btn_open_img = QPushButton(self)
    self.btn_open_img.setText("Open image")
    self.btn_open_img.move(10, 30)
    self.btn_open_img.clicked.connect(self.openimage)

    # add open popup window button
    self.btn_open_popup_window = QPushButton(self)
    self.btn_open_popup_window.setText("open popup window")
    self.btn_open_popup_window.move(10, 200)
    self.btn_open_popup_window.clicked.connect(self.open_popup_window)

    # add close popup window button
    self.btn_close_popup_window = QPushButton(self)
    self.btn_close_popup_window.setText("close popup window")
    self.btn_close_popup_window.move(10, 300)
    self.btn_close_popup_window.clicked.connect(self.close_popup_window)

  @staticmethod
  def open_popup_window():
    args.echo = True

  @staticmethod
  def close_popup_window():
    args.echo = False

  def openimage(self):
    img_name, _ = QFileDialog.getOpenFileName(self, "Open image", "", "*.jpg;;*.png;;All Files(*)")
    img = QtGui.QPixmap(img_name).scaled(args.image_size, args.image_size)
    self.label.setPixmap(img)
    text, prob = classifier(img_name)
    print("------------------------------")
    print(f"Label: {text:<75}")
    print(f"Probability: {prob:.6f}.")
    print("------------------------------")
    if args.echo:
      self.echo(str(text), prob)

  def echo(self, text, prob):
    QMessageBox.information(
      self, "Message",
      f"Label :{str(text):<75}\nProbability: {prob:.6f}")


if __name__ == "__main__":
  model = VGG.from_pretrained(args.model_name)
  model.eval()
  if args.image_size is None:
    args.image_size = VGG.get_image_size(args.model_name)
  # Preprocess image
  tfms = transforms.Compose([
    transforms.Resize(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  # Load class names
  labels_map = json.load(open(args.labels_map))
  labels_map = [labels_map[str(i)] for i in range(args.num_classes)]

  app = QtWidgets.QApplication(sys.argv)
  my = Picture()
  my.show()
  sys.exit(app.exec_())
