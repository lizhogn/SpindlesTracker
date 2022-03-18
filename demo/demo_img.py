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
import argparse

from module.preprocessing.img_process import img_preprocess
from module.detection.onnx_inference import YoloxInference
from module.skeleton.skeletonize import skeletonize_spindle
from module.mot.sort import det_association

def args_get():
    parser = argparse.ArgumentParser(description='Image/Images Prediction')
    parser.add_argument('--images', '-i', type=str,
                        help='images path or image dir')
    parser.add_argument('--model', '-m', type=str,
                        default="module/detection/weight/yolox.onnx",
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()

    return args

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

def object_detection(imgs_list, model_path):
    from module.detection import utils
    det_res = []
    masks = []
    yolox_infer = YoloxInference(model=model_path)

    raw_det = {}
    for idx, img_path in tqdm(enumerate(imgs_list), desc="progress of detecting:"):
        img = cv2.imread(img_path)
        bboxes, mask = yolox_infer.forward(img)
        points = utils.convert_mask_to_points(mask, up_factor=1)
        # save the bboxes and points to json
        raw_det[idx] = {
            "img_id": idx,
            "bboxes": bboxes.tolist(),
            "points": points.tolist()
        }
        # match the points that belong to the same bbox
        matched_bboxes, matched_points = utils.match_points_to_bboxes(bboxes, points)
        det_res.append({
            "img_id": idx,
            "bboxes": matched_bboxes,
            "points": matched_points
        })
        
        img_dir, img_base = os.path.split(img_path)

        # save the detection
        det_path = os.path.join(img_dir, "det_" + img_base)
        det_img = vis_bbox(img, bboxes)
        plt.imsave(det_path, det_img)
        
        # save the mask
        h, w, _ = img.shape
        mask = mixup(img, mask[:h, :w], 0.6, 0.4)
        mask_path = os.path.join(img_dir, "mask_" + img_base)
        plt.imsave(mask_path, mask)
        masks.append(mask)

    return det_res, masks

def extract_skeleton(imgs_path, det_res, masks):
    skeletons = []
    for i, img_path in tqdm(enumerate(imgs_path), desc="progress of skeletonizing:"):
        # load image
        img = cv2.imread(img_path)
        skeleton_img = skeletonize_spindle(img, det_res[i])
        skeletons.append(skeleton_img)
        idx, idy = np.nonzero(skeleton_img)

        mask = masks[i][:, :, ::-1]
        mask[idx, idy, :] = 255

        imgdir, imgname = os.path.split(img_path)
        skeleton_save = os.path.join(imgdir, "skeleton_"+imgname)
        cv2.imwrite(skeleton_save, mask)

    return skeletons

if __name__ == "__main__":
    
    args = args_get()
    input_path = args.images
    model_path = args.model
    
    # step0: Image Preprocessing
    if os.path.isdir(input_path):
        imgs = []
        for img_path in glob.glob(os.path.join(input_path, "frame*.PNG")):
            imgs.append(img_path)
    else:
        imgs = [input_path]

    # step1: Object Detection
    logger.info("Object Detection")
    det_res, masks = object_detection(imgs, model_path)

    # step2: Skeleton Extraction
    skeletons = extract_skeleton(imgs, det_res, masks)
    




