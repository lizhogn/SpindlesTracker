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
    return det_res

def extract_skeleton(imgs, det_res):
    imgs_num, img_h, img_w, img_c = imgs.shape
    skeleton_imgs = np.zeros(shape=(imgs_num, img_h, img_w), dtype=np.uint8)
    for i in tqdm(range(imgs_num), desc="progress of skeletonizing:"):
        cur_img = imgs[i]
        cur_det_info = det_res[i]
        skeleton_imgs[i] = skeletonize_spindle(cur_img, cur_det_info)
    return skeleton_imgs

def data_linking(det_res):
    mot_res = det_association(det_res)
    return mot_res

def tracking_skeleton_visualize(imgs, skeleton_imgs=None, mot_res=None):
    from module.utils.visualize import vis_mot_res, vis_skeleton_res
    from module.utils.imgs_convert import imgs2video
    
    imgs_res = imgs.copy()
    # tracking result
    if mot_res is not None:
        imgs_res = vis_mot_res(imgs_res, mot_res)
    
    # skeleton vis
    if skeleton_imgs is not None:
        imgs_res = vis_skeleton_res(imgs_res, skeleton_imgs)

    # save to the video
    save_path = "demo_output/tracking_res_pred.avi"
    imgs2video(imgs_res, save_path, fps=5)
    logger.info("save video to {}".format(save_path))
    

if __name__ == "__main__":
    
    input_path = "/home/zhognli/SpindleTracker/dataset/demo_data/MAX_XY point 3.tif"
    
    # step0: Image Preprocessing
    logger.info("Image Preprocessing")
    imgs = img_preprocess(input_path)

    # step1: Object Detection
    logger.info("Object Detection")
    det_res = object_detection(imgs)

    # step2: Skeleton Extraction
    logger.info("Skeleton Extraction")
    skeleton_imgs = extract_skeleton(imgs, det_res)
    
    # step3: Multi-Object Tracking
    logger.info("Multi-Object Tracking")
    mot_res = data_linking(det_res)
    
    # Alterative: visualize tracking and skeletonizing results
    logger.info("Generate result..")
    tracking_skeleton_visualize(imgs, mot_res=mot_res, skeleton_imgs=skeleton_imgs)






