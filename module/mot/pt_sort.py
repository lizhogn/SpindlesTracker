from __future__ import print_function

import os
import json
from tracemalloc import start
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from tqdm import tqdm
import tifffile
import cv2

import glob
import time
import random
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def elulerian_batch(point_dt, point_gt):
    """
        dt = np.random.randint(0, 10, (4, 2))
        gt = np.random.randint(0, 10, (5, 2))
        res = elulerian_batch(dt, gt)
    """
    point_dt = np.expand_dims(point_dt, 1)
    point_gt = np.expand_dims(point_gt, 0)
    delta_x = point_dt[..., 0] - point_gt[..., 0]
    delta_y = point_dt[..., 1] - point_gt[..., 1]
    dis_matrix = np.sqrt(delta_x**2 + delta_y**2)
    
    return dis_matrix

def associate_detections_to_trackers(detections, trackers, dis_threshold=10):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,2),dtype=int)

    dis_matrix = elulerian_batch(detections, trackers)

    if min(dis_matrix.shape) > 0:
        # a = (dis_matrix > dis_threshold).astype(np.int32)
        matched_indices = linear_assignment(dis_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(dis_matrix[m[0], m[1]] > dis_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self, point, time):
        """
        Initialises a tracker using initial bounding box.
        """

        KalmanBoxTracker.count += 1

        #define constant velocity model
        self.kf = KalmanFilter(dim_x=4, dim_z=2) 
        self.kf.F = np.array([[1,0,1,0],
                            [0,1,0,1],
                            [0,0,1,0],
                            [0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0],
                            [0,1,0,0]])

        # self.kf.R[2:,2:] *= 10.
        self.kf.P[2:,2:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[2:,2:] *= 0.01

        self.kf.x[:2] = point.reshape((2, 1))
        self.time_since_update = 0

        self.id = KalmanBoxTracker.count
        
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.trajectory = [[time, int(point[0]), int(point[1])]]
        self.pred_trajectory = [[time, int(point[0]), int(point[1])]]
        self.branches = []

    def update(self, point, time):
        """
        Updates the state vector with observed point.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(point.reshape(2, 1))
        self.trajectory.append([time, int(point[0]), int(point[1])])

    def predict(self, time):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        self.pred_trajectory.append([time, self.kf.x[0, 0], self.kf.x[1, 0]])
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x[:2, 0]

class PointSORT(object):
    
    def __init__(self, max_age=5, min_hits=3, dis_threshold=20):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dis_threshold = dis_threshold
        self.trackers = []
        self.trackers_keep = []
        
        self.frame_count = 0
    
    def trackers_predict(self, time):
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        ret = []
        flag_merged = False
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict(time)
            trk[:] = np.array([pos[0, 0], pos[1, 0]])
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        return trks
    
    def update(self, dets, time):
        if len(dets) == 0:
            dets = np.empty((0, 5))
        self.frame_count = 1

        # get the predicted locations from existing trackers.
        trks = self.trackers_predict(time)

        # assignment the dets with trks
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.dis_threshold)
        
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], time)
        
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i], time)
            self.trackers.append(trk)
        
        return dets, trks
    
    def get_trajectory(self, trk):
        if len(trk.branches) == 0:
            return trk.trajectory
        else:
            trajectory = trk.trajectory
            for b in trk.branches:
                trajectory.extend(self.get_trajectory(b))
            return trajectory
    
    def combine_trajectory(self):
        self.trackers.extend(self.trackers_keep)
        for trk in self.trackers:
            trajectory = self.get_trajectory(trk)
            # sort by time
            trk.trajectory = sorted(trajectory, key=lambda x: x[0])

def missing_pts_predict(tracker1, tracker2):
    pts_1 = np.array(tracker1.trajectory)
    pts_2 = np.array(tracker2.trajectory)
    start = max(pts_1[0, 0], pts_2[0, 0])
    end = max(pts_1[-1, 0], pts_2[-1, 0])
    for i, t in enumerate(range(start, end+1)):
        if t not in pts_1[:, 0]:
            # get the value from pred
            pred_i = tracker1.pred_trajectory[i]
            # pred_i = [t, *pts_1[i-1, 1:]]
            pts_1 = np.insert(pts_1, i, pred_i, 0)
        if t not in pts_2[:, 0]:
            # get the value from pred
            pred_i = tracker2.pred_trajectory[i]
            # pred_i = [t, *pts_2[i-1, 1:]]
            pts_2 = np.insert(pts_2, i, pred_i, 0)
    
    # crop the pts
    pts_1 = pts_1[pts_1[:, 0] >= start]
    pts_1 = pts_1[pts_1[:, 0] <= end]
    pts_2 = pts_2[pts_2[:, 0] >= start]
    pts_2 = pts_2[pts_2[:, 0] <= end]
        
    return pts_1, pts_2

def pt_filter(pts):
    # inital sort
    pt_sort = PointSORT(max_age=5, min_hits=3, dis_threshold=50)
    len_pt = len(pts)
    
    for i, pt in enumerate(pts):
        pt = np.array(pt)
        pt_sort.update(pt, i)
    
    trackers = pt_sort.trackers
        
    if len(trackers) > 2:
        # get the longest tracker
        trackers = sorted(trackers, key=lambda x: len(x.trajectory), reverse=True)
        trackers = trackers[:2]

    elif len(trackers) < 2:
        return [[], []]

    # pts_1 = trackers[0].trajectory
    # pts_2 = trackers[1].trajectory

    pts_1, pts_2 = missing_pts_predict(trackers[0], trackers[1])

    # start_ = pts_1[0, 0]
    # end_ = pts_1[-1, 0]
    
    # pts_1 = [*[[]]*start_, *pts_1.tolist(), *[[]]*(len_pt-end_-1)]
    # pts_2 = [*[[]]*start_, *pts_2.tolist(), *[[]]*(len_pt-end_-1)]

    return [pts_1, pts_2]


if __name__ == "__main__":
    import json

    mot_res = json.load(open("mot_res.json", "r"))
    for i in mot_res:
        pts = mot_res[i]["points"]
        pt_filter(pts)