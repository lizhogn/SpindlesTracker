import cv2
import numpy as np


def vis_skeleton_res(imgs, skeleton_imgs):
    for img, skeleton in zip(imgs, skeleton_imgs):
        idx, idy = np.nonzero(skeleton)
        img[idx, idy, 1] = 255
    return imgs

def vis_mot_res(imgs, mot_res):
    mot_imgs = imgs.copy()
    for mot_i in mot_res:
        det_infos = mot_res[mot_i]
        for det in det_infos["bboxes"]:
            frame, x1, y1, x2, y2 = [int(i) for i in det]
            cur_img = mot_imgs[frame, ...]
            cv2.rectangle(cur_img, 
                        pt1=(x1, y1), 
                        pt2=(x2, y2), 
                        color=(255, 255, 0), 
                        thickness=1)
            cv2.putText(cur_img, "id:{}".format(str(mot_i)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            mot_imgs[frame, ...] = cur_img
    return mot_imgs

def vis_mot_res1(imgs, mot_res):
    mot_imgs = imgs.copy()
    for mot_i in mot_res:
        det_infos = mot_res[mot_i]
        for det in det_infos["bboxes"]:
            frame, x1, y1, x2, y2 = [int(i) for i in det]
            cur_img = mot_imgs[frame, ...]
            cv2.rectangle(cur_img, 
                        pt1=(x1, y1), 
                        pt2=(x2, y2), 
                        color=(255, 255, 0), 
                        thickness=1)
            cv2.putText(cur_img, "id:{}".format(str(mot_i)), (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            mot_imgs[frame, ...] = cur_img
    return mot_imgs