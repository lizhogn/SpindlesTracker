"""
# predict one image demo
STEP 1: Object Detection
    Inputs:
        imgs: 3 channel imgs, shape is (N, H, W, 3)
    Returns:
        List of Dicts:
            Img_path: "path/to/img"
            Bboxes: [m, 4]
            Points: [m, 4]

STEP 2: Spindle Skeleton Extraction
    Inputs:
        Results of Detection: List of Dicts:
            Img_path: "path/to/img"
            Bboxes: [m, 4]
            Points: [m, 4]
    Returns:
        Skeleton imgs


"""
import numpy as np
from loguru import logger
from tqdm import tqdm
import sys, os
import glob
import cv2
import matplotlib.pyplot as plt

from module.preprocessing.img_process import img_preprocess
from module.detection.onnx_inference import YoloxInference
from module.skeleton.skeletonize import skeletonize_spindle
from module.mot.sort import det_association
from module.detection import utils

# step1: load model
yolox_infer = YoloxInference()

def predict(img, conf=0.1):
    
    # step2: object detection
    img = img[:, :, ::-1]       # change to BGR format
    bboxes, mask = yolox_infer.forward(img, conf_threshold=conf)
    points = utils.convert_mask_to_points(mask, up_factor=1)
    matched_bboxes, matched_points = utils.match_points_to_bboxes(bboxes, points)
    det_res = {"img_id": "0", "bboxes": matched_bboxes, "points": matched_points}

    # plot the points
    h, w, _ = img.shape
    img_mask = mixup(img.copy(), mask[:h, :w], 0.6, 0.4)

    # plot the skeleton
    skeleton_img = skeletonize_spindle(img.copy(), det_res)
    idx, idy = np.nonzero(skeleton_img)
    img_mask[idx, idy, :] = 255

    # save the detection
    det_img = vis_bbox(img_mask, bboxes)

    # cv2.imwrite("det.jpg", det_img)
    return det_img[:, :, ::-1]

def mixup(img, mask, rate_img=0.5, rate_mask=0.5):
    cmap = plt.get_cmap("jet")
    rgba_img = cmap(mask)
    rgb_img = np.delete(rgba_img, 3, 2) * 255
    return (img[:, :, ::-1] * rate_img + rgb_img * rate_mask).astype(np.uint8)


def vis_bbox(img, boxes):
    img_ = img[:, :, ::-1].copy()
    for i in range(len(boxes)):
        box = boxes[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = [0, 255, 0]
        cv2.rectangle(img_, (x0, y0), (x1, y1), color, 1)
        cv2.putText(img_, "id: {}".format(str(i)), (x0, y0-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img_

if __name__ == "__main__":
    img = cv2.imread("demo/data/img1.PNG")
    predict(img[:, :, ::-1])
    




