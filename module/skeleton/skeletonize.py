import numpy as np
from skimage import graph
from skimage import filters
from module.utils.imgs_convert import box_expand


def point_selector(bboxes, points):
    for i, (bx, pts) in enumerate(zip(bboxes, points)):
        x1, y1, x2, y2 = [int(x) for x in bx]
        pt_cnt = len(pts)
        if pt_cnt == 0:
            pt1 = [x1, y1]
            pt2 = [x2, y2]
            points[i] = [pt1, pt1]
        elif pt_cnt == 1:
            px, py = pts[0]
            # find the nearest vertex
            delta_x = [x1-px, x2-px]
            index_x = delta_x.index(min(delta_x))
            delta_y = [y1-py, y2-py]
            index_y = delta_y.index(min(delta_y))
            # get another pt
            pa = x1 if index_x == 1 else x2
            pb = y1 if index_y == 1 else y2

            points[i].append([round(pa), round(pb)])
        elif pt_cnt == 2:
            # do nothing
            pass
        elif pt_cnt > 2:
            # find 2 point that nearest to the vertex of box
            # calculate the distance
            pts = np.array(pts)
            vertex = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
            dist0 = np.power(pts - vertex[0], 2).sum(axis=1)
            dist1 = np.power(pts - vertex[1], 2).sum(axis=1)
            dist2 = np.power(pts - vertex[2], 2).sum(axis=1)
            dist3 = np.power(pts - vertex[3], 2).sum(axis=1)

            if dist0.min() + dist3.min() <= dist1.min() + dist2.min():
                # get 0 3 nearest points
                idx1 = np.argmin(dist0)
                idx2 = np.argmin(dist3)

            else:
                idx1 = np.argmin(dist1)
                idx2 = np.argmin(dist2)

            points[i] = [pts[idx1], pts[idx2]]
    return bboxes, points

def skeletonize_spindle(img, det_info):
    bboxes = det_info["bboxes"]
    points = det_info["points"]
    bboxes, points = point_selector(bboxes, points)
    cost_matrix = 255 - img[:, :, 2]
    h, w, _  = img.shape
    skeleton_img = np.zeros(shape=(h, w), dtype=np.uint8)
    
    for idx, (bx, pt) in enumerate(zip(bboxes, points), start=1):
        px1, py1 = pt[0]
        px2, py2 = pt[1]
        x1, y1, x2, y2 = box_expand(bx, (h, w), padding=15)
        start = [py1-y1, px1-x1]
        end = [py2-y1, px2-x1]
        crop_cost_matrix = cost_matrix[y1:y2, x1:x2]
        indices, weight = graph.route_through_array(crop_cost_matrix, start, end, fully_connected=True)
        indices = np.stack(indices, axis=-1)
        indices[0] += y1
        indices[1] += x1
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