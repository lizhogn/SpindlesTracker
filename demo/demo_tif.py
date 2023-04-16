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

STEP 2: Multi-Object Tracking
    Inputs:
        Results of Detection: List of Dicts:
            Img_path: "path/to/img"
            Bboxes: [m, 4]
            Points: [m, 4]
    Returns:
        Association results

STEP 3: Spindle Skeleton Extraction
    Inputs:
        Results of Detection: List of Dicts:
            Img_path: "path/to/img"
            Bboxes: [m, 4]
            Points: [m, 4]
    Returns:
        Skeleton imgs

"""

from tracemalloc import start
import numpy as np
from loguru import logger
from tqdm import tqdm
import json
import cv2
import argparse
import os

from module.preprocessing.img_process import img_preprocess
from module.detection.onnx_inference import YoloxInference
from module.skeleton.skeletonize import skeletonize_spindle
from module.mot.pt_sort import pt_filter
from module.mot.sort import det_association
from module.utils.bboxes import bboxes_filter


def args_get():
    parser = argparse.ArgumentParser(description='Image/Images Prediction')
    parser.add_argument('--images', '-i', type=str,
                        default="demo/data/demo_video.tif",
                        help='images(tif with green-red channel) path')
    parser.add_argument('--model', '-m', type=str,
                        default="module/detection/weight/yolox.onnx",
                        help='sum the integers (default: find the max)')
    parser.add_argument('--save_path', '-s', type=str,
                        default="./demo_output",
                        help="save path for the output file")

    args = parser.parse_args()

    return args

def object_detection(imgs, model=None, save_path=None):
    from module.detection import utils
    det_res = []
    yolox_infer = YoloxInference(model)

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

    with open(os.path.join(save_path, "raw_det.json"), "w") as f:
        json.dump(raw_det, f)
    with open(os.path.join(save_path, "points_filt_det.json"), "w") as f:
        json.dump(det_res, f)
    return det_res, raw_det

def extract_skeleton(imgs, det_res):
    imgs_num, img_h, img_w, img_c = imgs.shape
    skeleton_imgs = np.zeros(shape=(imgs_num, img_h, img_w), dtype=np.uint8)
    for i in tqdm(det_res, desc="progress of skeletonizing:"):
        cur_img = imgs[int(i)]
        cur_det_info = det_res[i]
        skeleton_imgs[int(i)] = skeletonize_spindle(cur_img, cur_det_info)
        
    return skeleton_imgs

def data_linking(det_res):
    mot_res = det_association(det_res)
    new_mot_res = {}
    for i in tqdm(mot_res, desc="progress of mot:"):
        cur_bboxes = np.asarray(mot_res[i]["bboxes"])
        cur_pts = mot_res[i]["points"]
        if len(cur_bboxes) <=2 or len(cur_pts) <=2:
            continue
        # bboxes filter
        time_start = cur_bboxes[0][0]
        cur_bboxes, cur_pts = bboxes_filter(cur_bboxes, cur_pts)
        
        # pt filter
        pts_1, pts_2 = pt_filter(cur_pts)
        if len(pts_1) != 0:
            pts_1[:, 0] = time_start + pts_1[:, 0]
            pts_2[:, 0] = time_start + pts_2[:, 0]

            new_mot_res[i] = {
                "bboxes": cur_bboxes,
                "pts_1": pts_1,
                "pts_2": pts_2
            }
    
    # mot to det
    new_det_res = {}
    for i in new_mot_res:
        cur_mot = new_mot_res[i]
        bboxes = cur_mot["bboxes"]
        pts_1 = cur_mot["pts_1"]
        pts_2 = cur_mot["pts_2"]
        for bx in bboxes:
            t =  int(bx[0])
            if t not in new_det_res:
                new_det_res[t] = {
                    "img_id": t,
                    "bboxes": [],
                    "points": []
                }
            new_det_res[t]["bboxes"].append(bx[1:].tolist())
            if t in pts_1[:, 0]:
                pt1 = pts_1[pts_1[:, 0]==t][0, 1:].tolist()
                pt2 = pts_2[pts_2[:, 0]==t][0, 1:].tolist()
                new_det_res[t]["points"].append([pt1, pt2])
            else:
                new_det_res[t]["points"].append([[], []])

    return new_mot_res, new_det_res

def tracking_skeleton_visualize(imgs, skeleton_imgs=None, mot_res=None, save_path=None):
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
    if save_path is not None:
        save_avi = os.path.join(save_path, "tracking_res_filter.avi")
        imgs2video(imgs_res, save_avi, fps=5)
        logger.info("save video to {}".format(save_avi))

def visualize_det_res(imgs, det_res):
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
    

    args = args_get()
    input_path = args.images
    model_path = args.model
    save_path = args.save_path

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # step0: Image Preprocessing
    logger.info("Image Preprocessing")
    imgs = img_preprocess(input_path)

    # step1: Object Detection
    logger.info("Object Detection")
    det_res, raw_res = object_detection(imgs, model=model_path, save_path=save_path)

    # load res
    # det_res = json.load(open("demo_output/points_filt_det.json", "r"))
    # raw_res = json.load(open("demo_output/raw_det.json", "r"))

    # step2: Multi-Object Tracking
    logger.info("Multi-Object Tracking")
    new_mot_res, new_det_res = data_linking(det_res)

    # step3: Skeleton Extraction
    logger.info("Skeleton Extraction")
    skeleton_imgs = extract_skeleton(imgs, new_det_res)

    # Alterative: visualize the raw object and keypoint detection results
    visualize_det_res(imgs, det_res=raw_res)

    # Alterative: visualize tracking and skeletonizing results
    logger.info("Generate result..")
    tracking_skeleton_visualize(imgs, mot_res=new_mot_res, skeleton_imgs=skeleton_imgs, save_path=save_path)