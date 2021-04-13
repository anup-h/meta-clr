import csv
import os
import re
import requests
import argparse
import subprocess
import multiprocessing
import pandas as pd  
import random


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

ratio = 0.0005
output_dir = 'youtube8m_rgb_raw/'
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
        if(m>200):
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
        limit = 50
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
            if k >= limit:
                break
        k = 0
        print("Downloading videos")
        for vid in video_ids_list:
            k = k + 1
            if (k % 50 == 0) or (k == limit):
                print(k)
            output_path = output_dir + vid + ".mp4"
            err, _ = download_video_using_youtube_dl(vid, output_path)
            if err:
                print("Video id: {} download unsuccessful.".format(vid))
