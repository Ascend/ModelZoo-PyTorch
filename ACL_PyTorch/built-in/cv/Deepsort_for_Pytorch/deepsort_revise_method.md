# deepsort源码适配Ascend NPU修改

## darknet.py文件修改

将darknet.py中Darknet类的forward函数返回值修改为。

~~~python
return out_boxes[0]['x'], out_boxes[0]['a'], out_boxes[1]['x'], out_boxes[1]['a'], out_boxes[2]['x'], out_boxes[2]['a']
~~~

## detector.py文件修改：

```
from .nms import boxes_nms
from .acl_net_dynamic import NetDynamic
```

YOLOv3类__init__函数中添加：

~~~python
self.dims = {'dimCount':4, 'name':'', 'dims':[1,3,416,416]}
self.model = NetDynamic(device_id = 0, model_path = "yolov3-sim.om")
self.anchors = [torch.tensor([3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]),
                torch.tensor([1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375]),
                torch.tensor([1.25, 1.625, 2.0, 3.75, 4.125, 2.875])]
~~~

__call__函数中with.no_grad()修改：

```
        with torch.no_grad():
            #img = img.to(self.device)
            #out_boxes = self.net(img)

            img = np.array(img, np.float32)
            img = np.ascontiguousarray(img, dtype = np.float32)
            out_boxes = self.model(img, self.dims)
            boxes = get_all_boxes(out_boxes, self.anchors, self.conf_thresh, self.num_classes,
                                  use_cuda=self.use_cuda)  # batch size is 1
```

## feature_extractor.py文件修改：

```
from .acl_net_dynamic import NetDynamic
```

类Extractor中__init__函数添加：

```
self.model = NetDynamic(device_id = 0, model_path = "deep_dims.om")
```

__call__函数修改为：

~~~python
def __call__(self, im_crops):
    im_batch = self._preprocess(im_crops)
    with torch.no_grad():
        im_batch = im_batch.to(self.device)
        dynamic_dim = im_batch.shape[0]
        dims = {'dimCount':4, 'name': '', 'dims': [dynamic_dim, 3, 128, 64]}
        im_batch = im_batch.cpu().numpy()
        features = self.model([im_batch], dims)
        #features = self.net(im_batch)
        return features[0]
def __del__(self):
    del self.model
~~~

## yolo_utils.py文件修改

get_all_boxes方法修改为。

~~~python
def get_all_boxes(output, output_anchors, conf_thresh, num_classes, only_objectness=1, validation=False, use_cuda=True):
    # total number of inputs (batch size)
    # first element (x) for first tuple (x, anchor_mask, num_anchor)
    batchsize = output[0]['x'].data.size(0)

    all_boxes = []
    for i in range(len(output)):
        #pred, anchors, num_anchors = output[i]['x'].data, output[i]['a'], output[i]['n'].item()
        pred, anchors, num_anchors = torch.from_numpy(output[i]), output_anchors[i], 3
        boxes = get_region_boxes(pred, conf_thresh, num_classes, anchors, num_anchors, \
                                 only_objectness=only_objectness, validation=validation, use_cuda=use_cuda)

        all_boxes.append(boxes)
    return torch.cat(all_boxes, dim=1)
~~~

