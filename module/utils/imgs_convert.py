# convert imgs to videos
import cv2
import numpy as np

def imgs2video(imgs, video_name, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    for img in imgs:
        video_writer.write(img.astype("uint8"))
    video_writer.release()