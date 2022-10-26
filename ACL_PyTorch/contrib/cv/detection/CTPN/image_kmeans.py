# Copyright 2021 Huawei Technologies Co., Ltd
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

#-*- coding:utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def image_kmeans(path):
    """[Cluster the width and height of the test image]

    Args:
        path ([str]): [images path]
    """
    imgs_list = os.listdir(path)
    imgs_shape_list = []
    for i in range(len(imgs_list)):
        img = cv2.imread(os.path.join(imgs_path, imgs_list[i]))
        h, w, c = img.shape
        rescale_fac = max(h, w) / 1000
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
        imgs_shape_list.append([h, w])
    imgs_shape_array = np.array(imgs_shape_list)

    cluster = KMeans(n_clusters=10)
    model = cluster.fit(imgs_shape_array)
    labels = model.labels_
    cluster_centers = model.cluster_centers_
    print(labels)
    print(cluster_centers)
    center_list = []
    for i in range(len(cluster_centers)):
        center_list.append([int(cluster_centers[i, 0]), int(cluster_centers[i, 1])])
    print(center_list)
    center_list_sort = sorted(center_list, key=(lambda x:x[0]))
    print(center_list_sort)
    class_count = []
    for i in range(len(cluster_centers)):
        class_count.append(0)
    color_use = ['r', 'g', 'b', 'm', 'c', 'y', 'r', 'g', 'b', 'm']
    color_list = []
    for i in range(len(labels)):
        class_count[labels[i]] = class_count[labels[i]] +1
        color_list.append(color_use[labels[i]])
    print(class_count)
    class_count_sort = []
    for i in range(len(cluster_centers)):
        class_count_sort.append(class_count[center_list.index(center_list_sort[i])])
    print(class_count_sort)

    # draw image
    plt.scatter(imgs_shape_array[:,1], imgs_shape_array[:, 0], c=color_list, marker='o')
    for i in range(len(cluster_centers)):
        plt.scatter(cluster_centers[i, 1], cluster_centers[i, 0], c='black', marker='*')
        plt.annotate(class_count[i], xy = (cluster_centers[i, 1], cluster_centers[i, 0]), 
        xytext = (cluster_centers[i, 1] + 0.1, cluster_centers[i, 0] + 0.1))
    plt.savefig('kmeans_10.png')


if __name__ == '__main__':
    imgs_path = './data/Challenge2_Test_Task12_Images'
    image_kmeans(imgs_path)