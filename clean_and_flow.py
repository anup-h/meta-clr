import os
import cv2 as cv
import numpy as np
import multiprocessing
import json
import numpy as np


raw_root = "youtube8m_rgb_raw/"
flow_root = "youtube8m_flow/"
clean_root = "youtube8m_rgb/"
vids = os.listdir(raw_root)

def map_and_add_channels(flow, min_flow=-20, max_flow=20):
    empty_ch = np.zeros((flow.shape[0], flow.shape[1], 1))
    flow[flow < min_flow] = min_flow
    flow[flow > max_flow] = max_flow
    flow = (flow - min_flow) / (max_flow-min_flow)
    flow = (flow * 255).astype(np.uint8)
    flow_3ch = np.zeros((flow.shape[0], flow.shape[1], 3),dtype=np.uint8)
    flow_3ch[:,:,0:2] = flow
    return flow_3ch

def worker_task(v):
    if (v[:-3]+"npy" in os.listdir(flow_root)):
        print("Skipping", v)
        return
    clip = mp.VideoFileClip(raw_root+v, target_resolution=(128, None))
    clip = crop(clip, x_center = clip.w//2, y_center = clip.h//2, height = 128, width = 128)
    #clip.write_videofile(clean_root+v, logger=None, audio=False)
    
    print("Starting flow for ", v)

    duration = clip.reader.nframes
    f = clip.get_frame(0)
    prvs = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    result = []
    rgb = []

    for i in range(1, duration):
        f = clip.get_frame(i)
        cv.cvtColor(f, cv.COLOR_BGR2RGB)
        curr = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs,curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        rgb.append(f)
        result.append(map_and_add_channels(flow))
        prvs = curr
    rgb = np.stack(rgb)
    result = np.stack(result)
    np.save(flow_root+v[:-4], result)
    np.save(clean_root+v[:-4], rgb)
    print("Finished flow for ", v)
    return v

pool = multiprocessing.Pool(processes=8)
pool.map(worker_task, vids)

lendict = {}
root = 'youtube8m_rgb/'
for i, v in enumerate(os.listdir(root)):
    clip = np.load(root+v)#mp.VideoFileClip(root+v)
    lendict[v[:-4]] = clip.shape[0]
with open('durations.json', 'w') as f:
    json.dump(lendict, f)
