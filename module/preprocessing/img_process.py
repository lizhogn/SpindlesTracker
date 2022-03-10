import cv2
import tifffile
import os
from tqdm import tqdm
import numpy as np

from .img_enhancement import brightness_adjust

def img_preprocess(tif_path:str):
    assert tif_path.endswith("tif"), \
        "except tif file format but give the {}".format(os.path.extsep(tif_path)[-1])

    # load the image
    images = tifffile.imread(tif_path) # [bz, c, w, h]
    b, c, w, h = images.shape

    res_imgs = np.zeros((b, w, h, 3), dtype=np.uint8) 
    # channel maybe 2(red and green) or 1(red)
    for i in tqdm(range(0, b)):
       
        if c == 2:
            img_g = images[i, 0, :, :]
            img_r = images[i, 1, :, :]
            adj_r = brightness_adjust(img_r, 0, 0.5) * 255
            adj_g = brightness_adjust(img_g, 0.75) * 255
        elif c == 1:
            img_r = images[i, 0, :, :]
            adj_r = brightness_adjust(img_r, 0, 0.5) * 255
            adj_g = np.zeros_like(adj_r)
        else:
            raise("the input image channels should be 2 or 1")

        img_b = np.zeros_like(adj_r)

        img_aug = np.stack([img_b, adj_g, adj_r], axis=2).astype(np.uint8)
       
        res_imgs[i, ...] = img_aug

    return res_imgs

if __name__ == "__main__":
    tif_path = "dataset/demo_data/MAX_XY point 3.tif"
    imgs_aug = img_preprocess(tif_path)
    
    # show res
    import cv2
    b, c, w, h = imgs_aug.shape
    for i in range(b):
        cur_img = imgs_aug[i, ::-1, :, :]
        cv2.imwrite("test.png", cur_img)
    