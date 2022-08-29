# mmdetection源码适配Ascend NPU修改

## bbox_nms.py文件修改

注册NPU算子BatchNMSOp。

在源文件中添加如下代码，这里的forward只是为了推导shape写的伪实现，只需要成功导出到onnx即可。

~~~python
class BatchNMSOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
        """
        boxes (torch.Tensor): boxes in shape (batch, N, C, 4).
        scores (torch.Tensor): scores in shape (batch, N, C).
        return:
            nmsed_boxes: (1, N, 4)
            nmsed_scores: (1, N)
            nmsed_classes: (1, N)
            nmsed_num: (1,)
        """

        # Phony implementation for onnx export
        nmsed_boxes = bboxes[:, :max_total_size, 0, :]
        nmsed_scores = scores[:, :max_total_size, 0]
        nmsed_classes = torch.arange(max_total_size, dtype=torch.long)
        nmsed_num = torch.Tensor([max_total_size])

        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num

    @staticmethod
    def symbolic(g, bboxes, scores, score_thr, iou_thr, max_size_p_class, max_t_size):
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = g.op('BatchMultiClassNMS',
            bboxes, scores, score_threshold_f=score_thr, iou_threshold_f=iou_thr,
            max_size_per_class_i=max_size_p_class, max_total_size_i=max_t_size, outputs=4)
        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num
~~~

将BatchNMSOp算子包装到方法中，适配mmdet的调用方式。

~~~python
def batch_nms_op(bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
    """
    boxes (torch.Tensor): boxes in shape (N, 4).
    scores (torch.Tensor): scores in shape (N, ).
    """

    if bboxes.dtype == torch.float32:
        bboxes = bboxes.reshape(1, 1000, 80, 4).half()
        scores = scores.reshape(1, 1000, 80).half()
    else:
        bboxes = bboxes.reshape(1, 1000, 80, 4)
        scores = scores.reshape(1, 1000, 80)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = BatchNMSOp.apply(bboxes, scores,score_threshold, iou_threshold, max_size_per_class, max_total_size)
    # max_total_size num_bbox
    nmsed_boxes = nmsed_boxes.float()
    nmsed_scores = nmsed_scores.float()
    nmsed_classes = nmsed_classes.long()
    dets = torch.cat((nmsed_boxes.reshape((max_total_size, 4)), nmsed_scores.reshape((max_total_size, 1))), -1)
    labels = nmsed_classes.reshape((max_total_size, ))
    return dets, labels
~~~

等价替换expand算子，使导出的onnx更简洁。

导出onnx时注释掉原方法，调用BNMS方法等价替换。

~~~python
def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False):
    
    ...
    
     # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        # export expand operator to onnx more nicely
        if torch.onnx.is_in_onnx_export:
            bbox_shape_tensor = torch.ones(multi_scores.size(0), num_classes, 4)
            bboxes = multi_bboxes[:, None].expand_as(bbox_shape_tensor)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 4)
            
    ...
    
# npu
    if torch.onnx.is_in_onnx_export():
        dets, labels = batch_nms_op(bboxes, scores, score_thr, 
                                    nms_cfg.get("iou_threshold"), 
                                    max_num, max_num)
        return dets, labels
    
    ...
~~~

## rpn_head.py文件修改

注册BNMS算子和batch_nms_op方法。

~~~python
class BatchNMSOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
        """
        boxes (torch.Tensor): boxes in shape (batch, N, C, 4).
        scores (torch.Tensor): scores in shape (batch, N, C).
        return:
            nmsed_boxes: (1, N, 4)
            nmsed_scores: (1, N)
            nmsed_classes: (1, N)
            nmsed_num: (1,)
        """

        # Phony implementation for onnx export
        nmsed_boxes = bboxes[:, :max_total_size, 0, :]
        nmsed_scores = scores[:, :max_total_size, 0]
        nmsed_classes = torch.arange(max_total_size, dtype=torch.long)
        nmsed_num = torch.Tensor([max_total_size])

        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num

    @staticmethod
    def symbolic(g, bboxes, scores, score_thr, iou_thr, max_size_p_class, max_t_size):
        nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = g.op('BatchMultiClassNMS',
            bboxes, scores, score_threshold_f=score_thr, iou_threshold_f=iou_thr,
            max_size_per_class_i=max_size_p_class, max_total_size_i=max_t_size, outputs=4)
        return nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num
        

def batch_nms_op(bboxes, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size):
    """
    boxes (torch.Tensor): boxes in shape (N, 4).
    scores (torch.Tensor): scores in shape (N, ).
    """

    if bboxes.dtype == torch.float32:
        bboxes = bboxes.reshape(1, 1000, 80, 4).half()
        scores = scores.reshape(1, 1000, 80).half()
    else:
        bboxes = bboxes.reshape(1, 1000, 80, 4)
        scores = scores.reshape(1, 1000, 80)

    nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = BatchNMSOp.apply(bboxes, scores,score_threshold, iou_threshold, max_size_per_class, max_total_size)
    # max_total_size num_bbox
    nmsed_boxes = nmsed_boxes.float()
    nmsed_scores = nmsed_scores.float()
    nmsed_classes = nmsed_classes.long()
    dets = torch.cat((nmsed_boxes.reshape((max_total_size, 4)), nmsed_scores.reshape((max_total_size, 1))), -1)
    labels = nmsed_classes.reshape((max_total_size, ))
    return dets, labels
~~~

导出onnx时注释掉原方法，调用BNMS方法等价替换。

~~~python
def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
   ...
   
    # npu return
        if torch.onnx.is_in_onnx_export():
            dets, labels = batch_nms_op(proposals, 
            							scores, 0.0, 
            							nms_cfg.get("iou_threshold"), 
            							cfg.nms_post, 
            							cfg.nms_post)
            return dets
        # cpu and gpu return
        else:
            dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
            return dets[:cfg.nms_post]
~~~

## single_level_roi_extractor.py文件修改

注册NPU RoiExtractor算子。

在源文件中添加如下代码，这里的forward只是为了推导shape写的伪实现，只需要成功导出到onnx即可。

~~~python
class RoiExtractor(torch.autograd.Function):
    @staticmethod
    def forward(self, f0, f1, f2, f3, rois, aligned=1, finest_scale=56, pooled_height=7, pooled_width=7,
                         pool_mode='avg', roi_scale_factor=0, sample_num=0, spatial_scale=[0.25, 0.125, 0.0625, 0.03125]):
        """
        feats (torch.Tensor): feats in shape (batch, 256, H, W).
        rois (torch.Tensor): rois in shape (k, 5).
        return:
            roi_feats (torch.Tensor): (k, 256, pooled_width, pooled_width)
        """

        # phony implementation for shape inference
        k = rois.size()[0]
        roi_feats = torch.ones(k, 256, pooled_height, pooled_width)
        return roi_feats

    @staticmethod
    def symbolic(g, f0, f1, f2, f3, rois):
        # TODO: support tensor list type for feats
        roi_feats = g.op('RoiExtractor', f0, f1, f2, f3, rois, aligned_i=1, finest_scale_i=56, pooled_height_i=7, pooled_width_i=7,
                         pool_mode_s='avg', roi_scale_factor_i=0, sample_num_i=0, spatial_scale_f=[0.25, 0.125, 0.0625, 0.03125], outputs=1)
        return roi_feats
~~~

在原forward方法中插入分支，导出onnx时使用RoiExtractor算子等价替换。

~~~python
@force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        # Work around to export onnx for npu
        if torch.onnx.is_in_onnx_export():
            roi_feats = RoiExtractor.apply(feats[0], feats[1], feats[2], feats[3], rois)
            # roi_feats = RoiExtractor.apply(list(feats), rois)
            return roi_feats

        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(
                -1, self.out_channels * out_size[0] * out_size[1])
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
	...  
~~~

## delta_xywh_bbox_coder.py文件修改

添加onnx export分支，利用numpy()将means和std的shape固定下来。

修改坐标的轴顺序，使切片操作在NPU上效率更高，整网性能提升约7%。

~~~python
# fix shape for means and stds when exporting onnx
def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True):
    if torch.onnx.is_in_onnx_export():
        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1).numpy() // 4)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1).numpy() // 4)
    else:
        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 4)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 4)
        
...
        
    # improve gather performance on NPU
    if torch.onnx.is_in_onnx_export():
        rois_perf = rois.permute(1, 0)
        # Compute center of each roi
        px = ((rois_perf[0, :] + rois_perf[2, :]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois_perf[1, :] + rois_perf[3, :]) * 0.5).unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = (rois_perf[2, :] - rois_perf[0, :]).unsqueeze(1).expand_as(dw)
        ph = (rois_perf[3, :] - rois_perf[1, :]).unsqueeze(1).expand_as(dh)
    else:
        # Compute center of each roi
        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        # Compute width/height of each roi
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)
        
    ...
~~~



# 去掉pytorch2onnx的算子检查

由于NPU自定义算子在 onnx 中未定义，需要去掉pytorch2onnx的检查，否则会报错。

1. 通过pip show torch找到pytorch安装位置，比如/home/mmdet/lib/python3.7/site-packages。
2. 打开文件/home/mmdet/lib/python3.7/site-packages/torch/onnx/utils.py。
3. 搜索_check_onnx_proto(proto)，并注释该行。



