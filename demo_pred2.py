"""
STEP 0: Image Preprocessing
    Inputs: 
        imgs: shape is (N, H, W, 3)
    Returns:
        Enhancement imgs

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

STEP 3: Multi-Object Tracking
    Inputs:
        Results of Detection: List of Dicts:
            Img_path: "path/to/img"
            Bboxes: [m, 4]
            Points: [m, 4]
    Returns:
        Association results

"""
import numpy as np
from loguru import logger
from tqdm import tqdm
import json
import cv2

from module.preprocessing.img_process import img_preprocess
from module.detection.onnx_inference import YoloxInference
from module.skeleton.skeletonize import skeletonize_spindle
from module.mot.sort import det_association


def object_detection(imgs):
    from module.detection import utils
    det_res = []
    yolox_infer = YoloxInference()

    raw_det = {}
    for idx, img in tqdm(enumerate(imgs), desc="progress of detecting:"):
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

    # save intermediate results
    with open("demo_output/raw_det.json", "w") as f:
        json.dump(raw_det, f)
    with open("demo_output/points_filt_det.json", "w") as f:
        json.dump(det_res, f)
    return raw_det

def tracking_skeleton_visualize(imgs, det_res):
    from module.utils.visualize import vis_mot_res, vis_skeleton_res
    from module.utils.imgs_convert import imgs2video
    
    imgs_res = imgs.copy()
    for i in det_res:
        cur_img = imgs[int(i)]
        cur_det = det_res[i]

        # show bbox
        bboxes = cur_det["bboxes"]
        for bx in bboxes:
            x1, y1, x2, y2, _ = [int(x) for x in bx]
            cv2.rectangle(cur_img, 
                        pt1=(x1, y1), 
                        pt2=(x2, y2), 
                        color=(255, 255, 0), 
                        thickness=1)
        # show points
        points = cur_det["points"]
        for pt in points:
            x, y = pt
            cv2.line(cur_img, (x-2, y), (x+2, y), (0, 255, 0), 1)
            cv2.line(cur_img, (x, y-2), (x, y+2), (0, 255, 0), 1)
        
        imgs_res[int(i), ...] = cur_img
            
    # save to the video
    save_path = "demo_output/tracking_res_raw.avi"
    imgs2video(imgs_res, save_path, fps=5)
    logger.info("save video to {}".format(save_path))
    

if __name__ == "__main__":
    
    input_path = "/home/zhognli/SpindleTracker/dataset/demo_data/MAX_XY point 3.tif"
    
    # step0: Image Preprocessing
    logger.info("Image Preprocessing")
    imgs = img_preprocess(input_path)

    # step1: Object Detection
    # logger.info("Object Detection")
    # det_res = object_detection(imgs)
    det_res = json.load(open("demo_output/raw_det.json", "r"))

    # Alterative: visualize tracking and skeletonizing results
    logger.info("Generate result..")
    tracking_skeleton_visualize(imgs, det_res=det_res)






