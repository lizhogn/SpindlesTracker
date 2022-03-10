# image sampler from tif multi-channels
# author: lizhogn
# time: 2021-3-19

import tifffile
import cv2
import numpy as np
import os, re, random
from tqdm import tqdm
import string

def brightness_adjust(img, lmin=0, lmax=0.75):
    # input:
    #       img: ndarray image, shape as [w, h]
    #       lmin: lower bound
    #       lmax: upper bound
    # return:
    #       image_adjust: Image with adjusted brightness range
    img = np.asarray(img)
    img_max, img_min = img.max(), img.min()
    inter = img_max - img_min
    image_adjust = (img - (img_min + lmin*inter)) / (inter * (lmax - lmin)+1e-4)
    image_adjust = np.clip(image_adjust, 0, 1)

    return image_adjust

def hdr_adjust(img, lmin=0.05, lmax=0.95):
    """
    input:
    :param img: 2D-array image
    :param lmin:
    :param lmax:
    :return:
    """
    # step1: hist statistic
    hist, bins = np.histogram(img.flatten(), 65536, [0, 65536] )
    # step2: determine the threshold: img_min, img_max
    sum = 0
    flag = True
    for i in range(65536):
        sum += hist[i]
        if (sum >= lmin * img.size) and flag:
            img_min = i
            flag = False
        if sum >= lmax * img.size:
            img_max = i
            break

    # step3:
    inter = img_max - img_min
    image_adjust = (img - img_min) / inter
    image_adjust = np.clip(image_adjust, 0, 1)

    return image_adjust

def hist_equalize(img):
    # img normalize
    img_max, img_min = img.max(), img.min()
    img = (img - img_min) / (img_max - img_min)
    img = (img * 255).astype(np.uint8)
    img_eq = cv2.equalizeHist(img)

    return img_eq