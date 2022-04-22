import torch

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


class RetinaNetDetailPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """

    def __init__(
            self,
            pre_nms_thresh,
            pre_nms_top_n,
            nms_thresh,
            fpn_post_nms_top_n,
            min_size,
            box_coder=None,
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            box_coder (BoxCoder)
        """
        super(RetinaNetDetailPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size

        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder

    def forward_for_single_feature_map(self, anchors, box_cls, box_regression):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        device = box_cls.device
        N, _, H, W = box_cls.shape
        A = int(box_regression.size(1) / 4)
        C = int(box_cls.size(1) / A)

        # put in the same format as anchors
        box_cls = box_cls.view(N, -1, C, H, W).permute(0, 3, 4, 1, 2)
        box_cls = box_cls.reshape(N, -1, C)
        box_cls = box_cls.sigmoid()

        box_regression = box_regression.view(N, -1, 4, H, W)
        box_regression = box_regression.permute(0, 3, 4, 1, 2)
        box_regression = box_regression.reshape(N, -1, 4)

        num_anchors = A * H * W

        results = [[] for _ in range(N)]
        pre_nms_thresh = self.pre_nms_thresh
        candidate_inds = box_cls > self.pre_nms_thresh
        if candidate_inds.sum().item() == 0:
            return results

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        for batch_idx, (per_box_cls, per_box_regression, per_pre_nms_top_n, \
                        per_candidate_inds, per_anchors) in enumerate(zip(
            box_cls,
            box_regression,
            pre_nms_top_n,
            candidate_inds,
            anchors)):

            # Sort and select TopN
            per_box_cls = per_box_cls[per_candidate_inds]
            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]
            per_class += 1
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_box_loc = per_box_loc[top_k_indices]
                per_class = per_class[top_k_indices]

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", per_box_cls)
            boxlist.add_field("sparse_off", per_box_loc / 9)
            boxlist.add_field("sparse_anchor_idx", per_box_loc % 9)
            boxlist.add_field("sparse_anchors",
                              per_anchors.bbox[per_box_loc, :].view(-1, 4))
            boxlist.add_field("sparse_batch",
                              per_box_loc.clone().fill_(batch_idx))

            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results[batch_idx] = boxlist

        return results

    def forward(self, anchors, box_cls, box_regression, targets=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]

        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        num_levels = len(box_cls)
        anchors = list(zip(*anchors))
        for a, o, b in zip(anchors, box_cls, box_regression):
            sampled_boxes.append(self.forward_for_single_feature_map(a, o, b))

        for layer in range(len(sampled_boxes)):
            for sampled_boxes_per_image in sampled_boxes[layer]:
                sampled_boxes_per_image.add_field(
                    'sparse_layers',
                    sampled_boxes_per_image.get_field('labels').clone().fill_(layer)
                )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]

        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            if len(boxlists[i]) == 0:
                results.append([])
                continue

            scores = boxlists[i].get_field("scores")
            labels = boxlists[i].get_field("labels")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]
            result = []
            # skip the background
            for j in range(1, 81):
                inds = (labels == j).nonzero().view(-1)
                if len(inds) == 0:
                    continue

                boxlist_for_class = boxlist[inds]
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, self.nms_thresh,
                    score_field="scores"
                )
                num_labels = len(boxlist_for_class)
                result.append(boxlist_for_class)

            result = cat_boxlist(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)

        return results


def make_retinanet_detail_postprocessor(
        config, fpn_post_nms_top_n, rpn_box_coder):
    pre_nms_thresh = 0.05
    pre_nms_top_n = 1000
    nms_thresh = 0.4
    fpn_post_nms_top_n = fpn_post_nms_top_n
    min_size = 0

    # nms_thresh = config.MODEL.RPN.NMS_THRESH
    # min_size = config.MODEL.RPN.MIN_SIZE
    box_selector = RetinaNetDetailPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        box_coder=rpn_box_coder,
        min_size=min_size
    )
    return box_selector
