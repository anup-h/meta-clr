import os
import moviepy.editor as mp
from moviepy.video.fx.all import crop
import cv2 as cv
import numpy as np
import multiprocessing
import json


raw_root = "youtube8m_rgb_raw/"
flow_root = "youtube8m_flow/"
clean_root = "youtube8m_rgb/"
vids = os.listdir(raw_root)

def worker_task(v):
    if (v[:-3]+"npy" in os.listdir(flow_root)):
        print("Skipping", v)
        return
    clip = mp.VideoFileClip(raw_root+v, target_resolution=(128, None))
    clip = crop(clip, x_center = clip.w//2, y_center = clip.h//2, height = 128, width = 128)
    clip.write_videofile(clean_root+v, logger=None, audio=False)

    print("Starting flow for ", v)

    duration = clip.reader.nframes
    f = clip.get_frame(0)
    prvs = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    result = []

    for i in range(1, duration):
        f = clip.get_frame(i)
        if(i%2==0):
            curr = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs,curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            result.append(flow)
            prvs = curr

    result = np.stack(result)
    np.save(flow_root+v[:-4], result)
    print("Finished flow for ", v)
    return v

pool = multiprocessing.Pool(processes=4)
pool.map(worker_task, vids)

lendict = {}
root = 'youtube8m_rgb/'
for i, v in enumerate(os.listdir(root)):
    clip = mp.VideoFileClip(root+v)
    lendict[v[:-4]] = clip.reader.nframes
with open('durations.json', 'w') as f:
    json.dump(lendict, f)
