import numpy as np
from findmaxima2d import find_maxima, find_local_maxima 

def convert_mask_to_points(mask, up_factor=None):
    # filter out the low score pixel
    mask[mask < 60] = 0
    
    # find the local maxium point
    local_max = find_local_maxima(mask)
    y, x, out = find_maxima(mask, local_max, 10)

    points = np.stack([x, y], axis=1)
    if up_factor is not None:
        points = up_factor * points
    return points

def isInParRect(point, box):
    # expand the box
    x1, y1, x2, y2 = box
    x1 -= 10
    y1 -= 10
    x2 += 10
    y2 += 10
    x, y = point
    if (x < x1) or (x > x2) or (y < y1) or (y > y2):
        return False
    else:
        return True

def vis_box_points(bboxes, points):
    import matplotlib.pyplot as plt
    import cv2

    bboxes = np.uint32(bboxes)
    img = cv2.imread("/home/zhognli/SpindleTracker/img_raw.png")
    for bx in bboxes[:, :-1]:
        pt1, pt2 = bx[:2].tolist(), bx[2:].tolist()
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
    plt.figure()
    plt.imshow(img)
    plt.plot(points[:, 0], points[:, 1], "ro", markersize=2)
    plt.savefig("vis_box_points.png")
    
def match_points_to_bboxes(bboxes, points):
    matched_bboxes = []
    matched_points = []
    # vis the points and bboxes for debug
    # vis_box_points(bboxes, points)
    for bx in bboxes[:, :-1]:
        matched_bboxes.append(bx.tolist())
        inside_points = []
        for pt in points:
            if isInParRect(pt, bx):
                inside_points.append(pt.tolist())
        matched_points.append(inside_points)
                
    return matched_bboxes, matched_points
                
if __name__ == "__main__":
    # load image
    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread("mask.png")
    img = img[:, :, 0]
    points = convert_mask_to_points(img, up_factor=1)
    plt.imshow(img)
    plt.plot(points[:, 1], points[:, 0], "ro")
    plt.savefig("test.png")
    