import numpy as np
from skimage.feature import peak_local_max

def convert_mask_to_points(mask):
    points = peak_local_max(np.squeeze(mask), min_distance=5)
    return points[:, ::-1]

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

def match_points_to_bboxes(bboxes, points):
    match_dict = {}
    matched_bboxes = []
    matched_points = []
    for pt in points:
        for idx, bx in enumerate(bboxes[:, :-1]):
            if isInParRect(pt, bx):
                if idx not in match_dict:
                    match_dict[idx] = []
                match_dict[idx].append(pt)
    for idx in match_dict:
        pts = match_dict[idx]
        if len(pts) == 2:
            matched_bboxes.append(bboxes[idx])
            matched_points.append([*pts[0], *pts[1]])
    return matched_bboxes, matched_points
                
if __name__ == "__main__":
    # load image
    import cv2
    img = cv2.imread("mask.png")
    points = peak_local_max(img, min_distance=5)
    print(points)