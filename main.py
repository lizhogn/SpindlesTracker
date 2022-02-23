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

from module.preprocessing import img_process
from module.detection.yolox_infer import spindle_detect
from module.skeleton.skeletonize import skeletonize_spindle
from module.mot.sort import det_association

def image_preprocessing(input_path):
    # load imgs

    # preprocessing

    imgs = np.zeros(shape=(140, 590, 640, 3))
    return imgs

def object_detection(imgs):
    from module.detection import utils
    det_res = []

    from module.detection.yolox_infer import data_generate
    data_iter = data_generate()
    
    for idx, img in tqdm(enumerate(imgs), desc="progress of detecting:"):
        img, bboxes, mask = next(data_iter)
        imgs[idx, ...] = img
        # boxes, mask = spindle_detect(img)
        points = utils.convert_mask_to_points(mask)
        # match the points that belong to the same bbox
        matched_bboxes, matched_points = utils.match_points_to_bboxes(bboxes, points)
        det_res.append({
            "img_id": idx,
            "bboxes": matched_bboxes,
            "points": matched_points
        })
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

def tracking_skeleton_visualize(imgs, skeleton_imgs, mot_res):
    from module.utils.visualize import vis_mot_res, vis_skeleton_res
    from module.utils.imgs_convert import imgs2video
    
    # tracking result
    imgs_tracking = vis_mot_res(imgs, mot_res)
    
    # skeleton vis
    imgs_skeleton = vis_skeleton_res(imgs_tracking, skeleton_imgs)

    # save to the video
    save_path = "demo/tracking_res.avi"
    imgs2video(imgs_skeleton, save_path, fps=5)
    logger.info("save video to {}".format(save_path))
    

if __name__ == "__main__":
    
    input_path = ""
    
    # step0: Image Preprocessing
    logger.info("Image Preprocessing")
    imgs = image_preprocessing(input_path)

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
    tracking_skeleton_visualize(imgs, skeleton_imgs, mot_res)





