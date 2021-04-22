import csv
import os
import re
import requests
import argparse
import subprocess
import multiprocessing
import pandas as pd  
import random
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
    
def crop(im, size):
    h, w, _ = im.shape
    im = cv.resize(im, None, fx=size/min(w, h), fy=size/min(w, h))
    h, w, _ = im.shape
    return im[h//2-size//2:h//2+size//2, w//2-size//2:w//2+size//2, :]

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
flow_root = 'youtube_flow/'
rgb_root = 'youtube_rgb/'
with open(METADATA_CSV_FILE, "r") as mtd:
    kg_data = csv.DictReader(mtd)
    selected_categories = []
    video_nums = []
    kg_ids = []
    for row in kg_data:
        selected_categories.append(row['Name'])
        video_nums.append(row['TrainVideoCount'])
        kg_ids.append(row['KnowledgeGraphId'])
    m = 0
    for cls, N, kg_id in zip(selected_categories, video_nums, kg_ids):
        m = m + 1
        if(m>1):
            break
        print(str(m) + '/3862')
        print("Fetching videos for '{}' category".format(cls))
        url_kg = JS_FETCHER_BASE_URL+kg_id[3:]+".js"
        err1, js_data = make_get_request(url_kg)
        if err1:
            print("Error")
        js_data = eval(js_data.lstrip('p("'+kg_id[3:]+'",').rstrip(');'))
        tf_records_ids = js_data[2:]
        random.shuffle(tf_records_ids)
        tf_records_ids_first_two_chars = [i[:2] for i in tf_records_ids]
        
        limit = 5
        print("There are " + str(limit) + " videos identified in this category")
        video_ids_list = []
        k = 0

        for ids, two_char_ids in zip(tf_records_ids, tf_records_ids_first_two_chars):
            k = k+1
            url = VIDEO_ID_FETCHER_BASE_URL + two_char_ids + "/" + ids + ".js"
            err2, id_data = make_get_request(url)
            if err2:
                continue
            video_id = re.findall(r"[a-zA-Z0-9_-]{11}", id_data)[0]
            video_ids_list.append(video_id)
            
            print("Downloading video: ", video_id)
            output_path = raw_dir + video_id + ".mp4"
            err, _ = download_video_using_youtube_dl(video_id, output_path)
            if err:
                print("Video id: {} download unsuccessful.".format(vid))
            if k >= limit:
                break
                
        print("Computing flows")
        def worker_task(file):
            if(file[:-4]+'_0' in os.listdir(rgb_root)):
                print('skipping ', file[:-4])
                return
            raw_path = raw_dir + file
            
            cap = cv.VideoCapture(raw_path)
            ret, prev_frame = cap.read()
            prev_frame = crop(prev_frame, 128)

            curr_rgb = []
            curr_flow = []

            duration = 1
            min_frames_since_last= 900
            clip_len = 32
            frames_since_last_seg = 1
            threshold = 0.95
            clips_so_far=0

            while ret:
                ret, curr_frame = cap.read()
                if ret:
                    if frames_since_last_seg>clip_len and frames_since_last_seg<min_frames_since_last:
                        frames_since_last_seg+=1
                        continue
                    curr_frame = crop(curr_frame, 128)
                    if frames_since_last_seg>min_frames_since_last:
                        non_zero = np.count_nonzero(cv.absdiff(curr_frame, prev_frame))/(128*128*3)
                        if non_zero>threshold:
                            curr_rgb=[]
                            curr_flow=[]
                            frames_since_last_seg=1
                            prev_frame = curr_frame
                            continue
                    if frames_since_last_seg<=clip_len:
                        curr_rgb.append(cv.cvtColor(curr_frame, cv.COLOR_BGR2RGB))
                        prev_grey = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
                        curr_grey = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
                        flow = cv.calcOpticalFlowFarneback(prev_grey,curr_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        curr_flow.append(map_and_add_channels(flow))
                    if frames_since_last_seg==clip_len:
                        np.save(rgb_root+file[:-4]+'_'+str(clips_so_far), np.stack(curr_rgb))
                        np.save(flow_root+file[:-4]+'_'+str(clips_so_far), np.stack(curr_flow))
                        clips_so_far+=1
                    frames_since_last_seg+=1
                    prev_frame = curr_frame
                    duration+=1
#             os.remove(raw_path)
        pool = multiprocessing.Pool(processes=8)
        pool.map(worker_task, os.listdir(raw_dir))

