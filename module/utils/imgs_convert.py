# convert imgs to videos
import cv2
import numpy as np

def imgs2video(imgs, video_name, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
        video_writer.write(img.astype("uint8"))
    video_writer.release()

def box_expand(box, img_size, padding=5):
    x1, y1, x2, y2 = [round(x) for x in box]
    h, w = img_size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    return [x1, y1, x2, y2]