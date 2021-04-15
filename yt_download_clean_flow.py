import csv
import os
import re
import requests
import argparse
import subprocess
import multiprocessing
import pandas as pd  
import random
import moviepy.editor as mp
from moviepy.video.fx.all import crop
import cv2 as cv
import json
import numpy as np


METADATA_CSV_FILE = "vocabulary.csv"
JS_FETCHER_BASE_URL = "https://storage.googleapis.com/data.yt8m.org/2/j/v/"
VIDEO_ID_FETCHER_BASE_URL = "https://storage.googleapis.com/data.yt8m.org/2/j/i/"
METADATA_CSV_FILE = "vocabulary.csv"

def make_get_request(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except Exception as err:
        return True, err
    else:
        return False, resp.text
    
def map_and_add_channels(flow, min_flow=-20, max_flow=20):
    empty_ch = np.zeros((flow.shape[0], flow.shape[1], 1))
    flow[flow < min_flow] = min_flow
    flow[flow > max_flow] = max_flow
    flow = (flow - min_flow) / (max_flow-min_flow)
    flow = (flow * 255).astype(np.uint8)
    flow_3ch = np.zeros((flow.shape[0], flow.shape[1], 3),dtype=np.uint8)
    flow_3ch[:,:,0:2] = flow
    return flow_3ch

def download_video_using_youtube_dl(video_id, output_path):
    command = [
        "youtube-dl",
        "-f", "best",
        "http://www.youtube.com/watch?v="+str(video_id),
        "-o", output_path
    ]
    proc = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    subprocess_pid = proc.pid
    try:
        subprocess_out, subprocess_err = proc.communicate()
    except:
        return True, subprocess_err
    else:
        return False, ""

# ratio = 0.0005
raw_dir = 'youtube8m_rgb_raw/'
flow_clean_root = 'youtube_clean/'
# with open(METADATA_CSV_FILE, "r") as mtd:
#     kg_data = csv.DictReader(mtd)
#     selected_categories = []
#     video_nums = []
#     kg_ids = []
#     for row in kg_data:
#         selected_categories.append(row['Name'])
#         video_nums.append(row['TrainVideoCount'])
#         kg_ids.append(row['KnowledgeGraphId'])
#     m = 0
#     for cls, N, kg_id in zip(selected_categories, video_nums, kg_ids):
#         m = m + 1
#         if(m<20):
#             continue
#         if(m>100):
#             break
#         print(str(m) + '/3862')
#         print("Fetching videos for '{}' category".format(cls))
#         url_kg = JS_FETCHER_BASE_URL+kg_id[3:]+".js"
#         err1, js_data = make_get_request(url_kg)
#         if err1:
#             print("Error")
#         js_data = eval(js_data.lstrip('p("'+kg_id[3:]+'",').rstrip(');'))
#         tf_records_ids = js_data[2:]
#         random.shuffle(tf_records_ids)
#         tf_records_ids_first_two_chars = [i[:2] for i in tf_records_ids]
        
#         limit = 30
#         print("There are " + str(limit) + " videos identified in this category")
#         video_ids_list = []
#         k = 0

#         for ids, two_char_ids in zip(tf_records_ids, tf_records_ids_first_two_chars):
#             k = k+1
#             url = VIDEO_ID_FETCHER_BASE_URL + two_char_ids + "/" + ids + ".js"
#             err2, id_data = make_get_request(url)
#             if err2:
#                 continue
#             video_id = re.findall(r"[a-zA-Z0-9_-]{11}", id_data)[0]
#             video_ids_list.append(video_id)
            
#             print("Downloading video: ", video_id)
#             output_path = raw_dir + video_id + ".mp4"
#             err, _ = download_video_using_youtube_dl(video_id, output_path)
#             if err:
#                 print("Video id: {} download unsuccessful.".format(vid))
#             if k >= limit:
#                 break
                
#         print("Computing flows")
#         def worker_task(file):
#             if(file[:-4]+'.npz' in os.listdir(flow_clean_root)):
#                 print('skipping ', file[:-4])
#                 return
#             raw_path = raw_dir + file
#             try:
#                 clip = mp.VideoFileClip(raw_path, target_resolution=(128, None))
#             except:
#                 print('Failed ', file[:-4])
#                 return
#             clip = crop(clip, x_center = clip.w//2, y_center = clip.h//2, height = 128, width = 128)

#             print("Starting flow for ", file[:-4])

#             duration = clip.reader.nframes
#             f = clip.get_frame(0)
#             prvs = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
#             result = []
#             rgb = []

#             for i in range(1, duration):
#                 f = clip.get_frame(i)
#                 cv.cvtColor(f, cv.COLOR_BGR2RGB)
#                 curr = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
#                 flow = cv.calcOpticalFlowFarneback(prvs,curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#                 rgb.append(f)
#                 result.append(map_and_add_channels(flow))
#                 prvs = curr
#             rgb = np.stack(rgb)
#             result = np.stack(result)
#             np.savez_compressed(flow_clean_root+file[:-4], rgb=rgb, flow=result)
#             os.remove(raw_path)
#         pool = multiprocessing.Pool(processes=8)
#         pool.map(worker_task, os.listdir(raw_dir))

lendict = {}
root = flow_clean_root
arrs = os.listdir(root)
def get_len(v):
    print(v)
    try:
        clip = np.load(root+v)
    except:
        print('failed', v)
        os.remove(root+v)
        return (v[:-4], -1)
    return (v[:-4], clip['rgb'].shape[0])
pool = multiprocessing.Pool(processes=8)
res = pool.map(get_len, arrs)

for k, v in res:
    if(v!=-1):
        lendict[k] = v
    
with open('durations.json', 'w') as f:
    json.dump(lendict, f)
