# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding:utf-8 -*-
import numpy as np
import cv2
from ctpn.config import *

"""
anchor生成
遇到的问题：首先，base_anchor 为初始位置点生成的anchor，按步长在feature map 的各个点生成anchor之后，anchors的 shape 为[10, h*w, 4]。
这里，我一开始是直接将anchors reshape 成 [10*h*w, 4]，这在训练时不收敛。
原因浅析：按我代码的实现方式，直接[10, h*w, 4] -> [10*h*w, 4]，anchor 的排列顺序将按照不同的anchor形状（共10种）进行排列，而不是根据feature map 的点按序排列，
而按 ctpn 的实现方式，小的anchor需要连成大的文本框才是最终的结果，不按点的顺序生成anchor可能给训练带来较大的干扰。
解决方案：将 anchor 根据feature_map 的各个点，按序生成10个anchor重新排列，也即：[10, h*w, 4] -> [h*w, 10, 4] -> [10*h*w, 4]，问题解决。
"""


def gen_anchor(featuresize, scale,
               heights=[11, 16, 23, 33, 48, 68, 97, 139, 198, 283],
               widths=[16, 16, 16, 16, 16, 16, 16, 16, 16, 16]):
    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel()), axis=1)

    # base center(x,,y) -> (x1, y1, x2, y2)
    base_anchor = np.array([0, 0, 15, 15])
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5
    #    heights = np.array(heights).view(len(heights), 1)
    #    widths = np.array(widths).view(len(widths), 1)

    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    anchors = list()
    for i in range(base_anchor.shape[0]):
        anchor_x1 = shift[:, 0] + base_anchor[i][0]
        anchor_y1 = shift[:, 1] + base_anchor[i][1]
        anchor_x2 = shift[:, 2] + base_anchor[i][2]
        anchor_y2 = shift[:, 3] + base_anchor[i][3]
        anchor_.append(np.dstack((anchor_x1, anchor_y1, anchor_x2, anchor_y2)))

    #    return np.squeeze(np.array(anchor)).transpose((1,0,2)).view((-1, 4))
    return np.squeeze(np.array(anchors)).transpose((1, 0, 2)).reshape((-1, 4))


"""
anchor 与 bbox的 iou计算
iou = inter_area/(bb_area + anchor_area - inter_area)
"""


def compute_iou(anchors, bbox):
    ious = np.zeros((len(anchors), len(bbox)), dtype=np.float32)
    anchor_area = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    for num, _bbox in enumerate(bbox):
        bb = np.tile(_bbox, (len(anchors), 1))
        bb_area = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
        inter_h = np.maximum(np.minimum(bb[:, 3], anchors[:, 3]) - np.maximum(bb[:, 1], anchors[:, 1]), 0)
        inter_w = np.maximum(np.minimum(bb[:, 2], anchors[:, 2]) - np.maximum(bb[:, 0], anchors[:, 0]), 0)
        inter_area = inter_h * inter_w
        area = bb_area + anchor_area - inter_area
        for i in range(len(bb_area)):
            area[i] = abs(bb_area[i]) + abs(anchor_area[i]) - inter_area[i]
        ious[:, num] = inter_area / area

    return ious


"""
计算 anchor与 gtboxes在垂直方向的差异参数 regression_factor(Vc, Vh)
1、(x1, y1, x2, y2) -> (ctr_x, ctr_y, w, h)
2、 Vc = (gt_y - anchor_y) / anchor_h
    Vh = np.log(gt_h / anchor_h)
"""


def bbox_transfrom(anchors, gtboxes):
    gt_y = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    gt_h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0

    anchor_y = (anchors[:, 1] + anchors[:, 3]) * 0.5
    anchor_h = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (gt_y - anchor_y) / anchor_h
    Vh = np.log(gt_h / anchor_h)

    return np.vstack((Vc, Vh)).transpose()


"""
已知 anchor和差异参数 regression_factor(Vc, Vh),计算目标框 bbox
"""


def transform_bbox(anchors, regression_factor):
    anchor_y = (anchors[:, 1] + anchors[:, 3]) * 0.5
    anchor_x = (anchors[:, 0] + anchors[:, 2]) * 0.5
    anchor_h = anchors[:, 3] - anchors[:, 1] + 1

    Vc = regression_factor[0, :, 0]
    Vh = regression_factor[0, :, 1]

    bbox_y = Vc * anchor_h + anchor_y
    bbox_h = np.exp(Vh) * anchor_h

    x1 = anchor_x - 16 * 0.5
    y1 = bbox_y - bbox_h * 0.5
    x2 = anchor_x + 16 * 0.5
    y2 = bbox_y + bbox_h * 0.5
    bbox = np.vstack((x1, y1, x2, y2)).transpose()

    return bbox


"""
bbox 边界裁剪
    x1 >= 0
    y1 >= 0
    x2 < im_shape[1]
    y2 < im_shape[0]
"""


def clip_bbox(bbox, im_shape):
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


"""
bbox尺寸过滤，舍弃小于设定最小尺寸的bbox
"""


def filter_bbox(bbox, minsize):
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep


"""
RPN module
1、生成anchor
2、计算anchor 和真值框 gtboxes的 iou
3、根据 iou，给每个anchor分配标签，0为负样本，1为正样本，-1为舍弃项
    (1) 对每个真值框 bbox，找出与其 iou最大的 anchor，设为正样本
    (2) 对每个anchor，记录其与每个bbox求取的 iou中最大的值 max_overlap
    (3) 对max_overlap 大于设定阈值的anchor，将其设为正样本，小于设定阈值，则设定为负样本
4、过滤超出边界的anchor框，将其标签设定为 -1
5、选取不超过设定数量的正负样本
6、求取anchor 取得max_overlap 时的gtbbox之间的真值差异量(Vc, Vh)
"""


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    base_anchor = gen_anchor(featuresize, scale)
    overlaps = compute_iou(base_anchor, gtboxes)

    gt_argmax_overlaps = overlaps.argmax(axis=0)
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)
    labels[gt_argmax_overlaps] = 1
    labels[anchor_max_overlaps > IOU_POSITIVE] = 1
    labels[anchor_max_overlaps < IOU_NEGATIVE] = 0

    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgsize[1]) |
        (base_anchor[:, 3] >= imgsize[0])
    )[0]
    labels[outside_anchor] = -1

    fg_index = np.where(labels == 1)[0]
    if (len(fg_index) > RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - RPN_POSITIVE_NUM, replace=False)] = -1
    if not OHEM:
        bg_index = np.where(labels == 0)[0]
        num_bg = RPN_TOTAL_NUM - np.sum(labels == 1)
        if (len(bg_index) > num_bg):
            labels[np.random.choice(bg_index, len(bg_index) - num_bg, replace=False)] = -1

    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets]


"""
非极大值抑制，去除重叠框
"""


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


"""
基于图的文本行构造算法
子图连接规则，根据图中配对的文本框生成文本行
1、遍历 graph 的行和列，寻找列全为false、行不全为false的行和列，索引号为index 
2、找到 graph 的第 index 行中为true的那项的索引号，加入子图中，并将索引号迭代给index
3、重复步骤2，直到 graph 的第 index 行全部为false
4、重复步骤1、2、3，遍历完graph
返回文本行list[文本框索引]
"""


class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)

        return sub_graphs


"""
配置参数
MAX_HORIZONTAL_GAP: 文本行内，文本框最大水平距离
MIN_V_OVERLAPS: 文本框最小垂直iou
MIN_SIZE_SIM: 文本框尺寸最小相似度
"""


class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MAX_HORIZONTAL_GAP = 60
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6


class TextProposalGraphBuilder:
    """
    构建配对的文本框
    """

    def __init__(self):
        self.im_size = im_size
        self.scores = scores
        self.text_proposals = text_proposals
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1
        self.boxes_table = boxes_table

    def get_successions(self, index):
        """
        遍历[x0, x0+MAX_HORIZONTAL_GAP]
        获取指定索引号的后继文本框
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 1, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def get_precursors(self, index):
        """
        遍历[x0-MAX_HORIZONTAL_GAP， x0]
        获取指定索引号的前驱文本框
        """
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 1, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results

        return results

    def is_succession_node(self, index, succession_index):
        """
        判断是否是配对的文本框
        """
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True

        return False

    def meet_v_iou(self, index1, index2):
        """
        判断两个文本框是否满足垂直方向的iou条件
        overlaps_v: 文本框垂直方向的iou计算。 iou_v = inv_y/min(h1, h2)
        size_similarity: 文本框在垂直方向的高度尺寸相似度。 sim = min(h1, h2)/max(h1, h2)
        """

        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        """
        根据文本框构建文本框对
        self.heights: 所有文本框的高度
        self.boxes_table: 将文本框根据左上点的x1坐标进行分组
        graph: bool类型的[n, n]数组，表示两个文本框是否配对，n为文本框的个数
            (1) 获取当前文本框Bi的后继文本框
            (2) 选取后继文本框中得分最高的，记为Bj
            (3) 获取Bj的前驱文本框
            (4) 如果Bj的前驱文本框中得分最高的恰好是 Bi，则<Bi, Bj>构成文本框对
        """

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            if self.is_succession_node(index, succession_index):
                graph[index, succession_index] = True

        return Graph(graph)


class TextProposalConnectorOriented:
    """
    连接文本框，构建文本行bbox
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        """
        将文本框连接起来，按文本行分组
        """
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)

        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        """
        一元线性函数拟合X，Y，返回y1, y2的坐标值
        """
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        根据文本框，构建文本行
        1、将文本框划分成文本行组，每个文本行组内包含符合规则的文本框
        2、处理每个文本行组，将其串成一个大的文本行
            (1) 获取文本行组内的所有文本框 text_line_boxes
            (2) 求取每个组内每个文本框的中心坐标 (X, Y)，最小、最大宽度坐标值 (x0 ,x1)
            (3) 拟合所有中心点直线 z1
            (4) 设置offset为文本框宽度的一半
            (5) 拟合组内所有文本框的左上角点直线，并返回当x取 (x0+offset, x1-offset)时的极作极右y坐标 （lt_y, rt_y）
            (6) 拟合组内所有文本框的左下角点直线，并返回当x取 (x0+offset, x1-offset)时的极作极右y坐标 （lb_y, rb_y）
            (7) 取文本行组内所有框的评分的均值，作为该文本行的分数
            (8) 生成文本行基本数据
        3、生成大文本框
        """
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)

        text_lines = np.zeros((len(tp_groups), 8), np.float32)
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]

            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2
            x0 = np.min(text_line_boxes[:, 0])
            x1 = np.max(text_line_boxes[:, 2])

            z1 = np.polyfit(X, Y, 1)

            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5

            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs


if __name__ == '__main__':
    anchor = gen_anchor((10, 15), 16)
