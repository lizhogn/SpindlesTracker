from sysconfig import get_path
import numpy as np
from tqdm import tqdm
import os
from module.mot.sort import Sort
import motmetrics as mm
import xml.etree.ElementTree as ET

def load_dataset(dataset_path, gt_path):
    root = ET.parse(dataset_path).getroot()
    det_res = []
    gt_text = []
    fp = open(gt_path, "w")
    for obj in root.iter("track"):
        track_id = int(obj.attrib["id"])
        track_label = obj.attrib["label"]
        if track_label == "microtubule":
            for fr in obj.iter("box"):
                frame_id = int(fr.attrib["frame"])
                x1 = float(fr.attrib["xtl"])
                y1 = float(fr.attrib["ytl"])
                x2 = float(fr.attrib["xbr"])
                y2 = float(fr.attrib["ybr"])
                det_res.append([frame_id, x1, y1, x2, y2, 1])
                gt_text.append("{},{},{:.2f},{:.2f},{:2f},{:.2f},{},-1,-1,-1".format(frame_id, track_id, x1, y1, x2-x1, y2-y1, 1))
    # sort by fame
    gt_text = sorted(gt_text, key=lambda x: int(x.split(",")[0]))
    for gt in gt_text:
        print(gt, file=fp)
    # sort by frame
    det_res = sorted(det_res, key=lambda x: x[0])
    det_res = np.stack(det_res, axis=0)
    fp.close()
    return det_res

def sort_mot(dataset, dt_path):
    # sort initial
    mot_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.5)
    num_frame = len(dataset)
    det_text = []
    # tracking
    with open(dt_path,'w') as out_file:
        for frame in tqdm(range(num_frame)):
            dets = dataset[dataset[:, 0]==frame, 1:6]
            trackers = mot_tracker.update(dets)
            for d in trackers:
                det_text.append('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))

        # sort by id
        det_text = sorted(det_text, key=lambda x: int(x.split(",")[1]))
        for dt in det_text:
            print(dt, file=out_file)


if __name__ == "__main__":
    # step1: load the dataset
    dataset_path = "/home/zhognli/YOLOX/datasets/total/total/annotations.xml"
    gt_path = "exps/tracking/gt.txt"
    dt_path = "exps/tracking/dt.txt"
    dataset = load_dataset(dataset_path, gt_path)

    # step2: call the sort
    sort_mot(dataset, dt_path)

    # step3: compare with the groudtruth
    gt=mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1, delim_whitespace=True)
    ts=mm.io.loadtxt(dt_path, fmt="mot15-2D")

    acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    mh = mm.metrics.create()
    metrics = list(mm.metrics.motchallenge_metrics)
    summary = mh.compute(acc, metrics=metrics, name="spindle")
    print(mm.io.render_summary(summary, formatters=mh.formatters,namemap=mm.io.motchallenge_metric_names))