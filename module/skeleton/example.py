import os
import pickle
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random
from skimage import filters
import matplotlib.pyplot as plt
from skimage import graph


def vis_bbox(img, boxes):

    for i in range(len(boxes)):
        box = boxes[i]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = [0, 255, 0]

        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
        cv2.putText(img, "id: {}".format(str(i)), (x0, y0-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img

def vis_heatmap(img, points):
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


def vis_skeleton(img, points, boxes):
    h, w, c  = img.shape
    skeleton_img = np.zeros(shape=(h, w), dtype=np.uint8)
    for i in range(len(boxes)):
        # find the points that in box
        box = boxes[i]
        points_in_bbox = find_points_in_box(points, box)
        cost = -img[:, :, 2]
        # minimum cost path
        if len(points_in_bbox) == 2:
            start = points_in_bbox[0][::-1]
            end = points_in_bbox[1][::-1]
            indices, weight =  graph.route_through_array(cost, start, end, fully_connected=True)
            indices = np.stack(indices, axis=-1)
            skeleton_img[indices[0], indices[1]] = 255
    return skeleton_img
            
def find_points_in_box(points, box):
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])
    points_in_bbox = []
    for point in points:
        if point[0] >= x0 and point[0] <= x1 and point[1] >= y0 and point[1] <= y1:
            points_in_bbox.append(point[:2])
        if point[2] >= x0 and point[2] <= x1 and point[3] >= y0 and point[3] <= y1:
            points_in_bbox.append(point[2:])
    return points_in_bbox

class SpindleVis():
    def __init__(self, img_dir, anno_path) -> None:
        self.img_dir = img_dir
        self.anno_path = anno_path

        # load img and annotation
        self.annotations = self.load_annotations()

    def load_annotations(self):
        """parser the xml annotation file"""
        # step1: load the xml file
        xml_root = ET.parse(self.anno_path).getroot()
        anno_list = []
        img_lost  = []
        for children in xml_root.iter("image"):
            print(children.tag, ":", children.attrib)
            img_attrib = children.attrib
            img_name = img_attrib["name"]
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                img_attrib["img_path"] = img_path
            else:
                img_lost.append(img_path)
                continue
            img_attrib["bboxes"] = []
            img_attrib["points"] = []
            for box in children.iter("box"):
                box_attrib = box.attrib
                img_attrib["bboxes"].append([
                    int(float(box_attrib["xtl"])),
                    int(float(box_attrib["ytl"])),
                    int(float(box_attrib["xbr"])),
                    int(float(box_attrib["ybr"]))
                ])

            for points in children.iter("points"):
                points_str = points.attrib["points"]
                p1, p2 = points_str.split(";")
                p1 = [int(float(p)) for p in p1.split(",")]
                p2 = [int(float(p)) for p in p2.split(",")]
                img_attrib["points"].append([*p1, *p2])
            anno_list.append(img_attrib)
        return anno_list
    
    def load_img(self, img_path):
        img = cv2.imread(img_path)
        return img

    def vis_sample(self, idx):
        img_attrib = self.annotations[idx]
        img = self.load_img(img_attrib["img_path"])
        bboxes = img_attrib["bboxes"]
        points = img_attrib["points"]
        # draw bbox for detection head
        img_det = vis_bbox(img, bboxes)
        cv2.imwrite("img_det.png", img_det)

        # draw points for heatmap head
        img_heatmap = vis_heatmap(img, points)
        cv2.imwrite("img_mask.png",img_heatmap)

    def draw_det_res(self):
        for i in range(len(self.annotations)):
            img_attrib = self.annotations[i]
            img = self.load_img(img_attrib["img_path"])
            bboxes = img_attrib["bboxes"]
            points = img_attrib["points"]
            # draw bbox for detection head
            img_det = vis_bbox(img.copy(), bboxes)
            cv2.imwrite("img_det_{}.png".format(i), img_det)

            # draw points for heatmap head
            img_heatmap = vis_heatmap(img, points)
            cmap = plt.get_cmap("jet")
            rgba_img = cmap(img_heatmap)
            rgb_img = np.delete(rgba_img, 3, 2) * 255
            
            # mixup img with heatmap
            img_mix = (img * 0.4 + rgb_img * 0.7).astype(np.uint8)
            cv2.imwrite("img_mix_{}.png".format(i), img_mix[:,:,::-1])

            # draw the skeleton of image
            img_skeleton = vis_skeleton(img, points, bboxes)
            img_skeleton_cmap = np.repeat(img_skeleton[:, :, np.newaxis], 3, axis=2)
            # mixup skeleton with heatmap
            img_heatmap_mix  = np.clip(img_skeleton_cmap * 1 + img * 0.7 + rgb_img[:, :, ::-1]*0.5, 0, 255).astype(np.uint8)
            cv2.imwrite("img_skeleton_{}.png".format(i), img_heatmap_mix)

            


if __name__ == "__main__":
    spindle_inst = SpindleVis(img_dir="./images", anno_path="annotations.xml")
    spindle_inst.draw_det_res()
    