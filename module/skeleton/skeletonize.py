import numpy as np
from skimage import graph
from skimage import filters


def skeletonize_spindle(img, det_info):
    bboxes = det_info["bboxes"]
    points = det_info["points"]
    cost_matrix = 255-img[:, :, 2]
    h, w, _  = img.shape
    skeleton_img = np.zeros(shape=(h, w), dtype=np.uint8)

    for idx, pt in enumerate(points, start=1):
        start = pt[:2][::-1]
        end = pt[2:][::-1]
        indices, weight = graph.route_through_array(cost_matrix, start, end, fully_connected=True)
        indices = np.stack(indices, axis=-1)
        skeleton_img[indices[0], indices[1]] = idx

    return skeleton_img

def vis_heapmap(img, points):
    h, w, c = img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    for i in range(len(points)):
        point = points[i]
        x0 = int(point[0])
        y0 = int(point[1])
        x1 = int(point[2])
        y1 = int(point[3])

        heatmap[y0, x0] = 1
        heatmap[y1, x1] = 1
    
    gauss_img = filters.gaussian(heatmap, sigma=4)
    gauss_img = gauss_img / gauss_img.max()
    return gauss_img