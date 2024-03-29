#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime
import torch
import requests

def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 0
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 0

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

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
        ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def demo_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def draw_detection(img, bboxes, heatmap):
    img = img.copy()
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = [int(x) for x in box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            "{:.2f}".format(box[4]),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    # mixup the heatmap and img
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap[heatmap<=60] = 0
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = 0.2 * np.float32(heatmap) + 0.8 * np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255*cam)

class YoloxInference(object):
    def __init__(self, model=None, input_shape=(640, 640), output_dir=None):
        self.input_shape = input_shape
        self.with_p6 = False
        self.score_thr = 0.3
        self.output_dir = output_dir
        if model is None:
            # download the model from huggingface
            model_path = "module/detection/weight/yolox.onnx"
            if not os.path.exists(model_path):
                model_url = "https://huggingface.co/lizhogn/YOLOX-SP/resolve/main/yolox.onnx"
                if not os.path.exists(os.path.dirname(model_path)):
                    os.makedirs(os.path.dirname(model_path))
                os.system("wget {} -O {}".format(model_url, model_path))
            self.model = model_path
        else:
            self.model = model
        
        # self.session = onnxruntime.InferenceSession(self.model, providers=['CUDAExecutionProvider'])
        self.session = onnxruntime.InferenceSession(self.model)
    
    def map_local_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward(self, origin_img, conf_threshold=0.1, visualize=False):
        if isinstance(origin_img, str):
            origin_img = cv2.imread(origin_img)
        img_h, img_w = origin_img.shape[:2]
        # model forward
        img, ratio = preprocess(origin_img, self.input_shape)
        ort_inputs = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.session.run(None, ort_inputs)

        # detection post-process
        detections = demo_postprocess(output[0], self.input_shape, p6=self.with_p6)[0]
        boxes = detections[:, :4]
        scores = detections[:, 4:5] * detections[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=conf_threshold)
        if dets is not None:
            final_boxes, final_cls_inds = dets[:, :5], dets[:, 5] 
        else:
            final_boxes, final_cls_inds = np.empty(shape=(0, 4)), np.empty(shape=(0, 1))
        
        # heatmap post-process
        heatmap = (output[1].squeeze()*255).astype(np.uint8)
        origin_shape = (int(self.input_shape[0] / ratio), int(self.input_shape[1] / ratio))
        heatmap = cv2.resize(heatmap, origin_shape)
        heatmap = heatmap[:img_h, :img_w]

        # v
        if visualize:
            vis_img = draw_detection(origin_img, final_boxes, heatmap)
            cv2.imwrite("demo/data/vis_img.png", vis_img)
            print("save the img result to demo/data/vis_img.png")
        return final_boxes, heatmap

if __name__ == '__main__':
    img_path = "demo/data/img1.PNG"

    yolox_infer = YoloxInference(model="module/detection/weight/yolox.onnx")
    dets, mask = yolox_infer.forward(img_path, visualize=True)
    
    print("done")
   
