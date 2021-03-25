import cv2
import torchvision
import numpy as np
import torch
import os
import json
import random

class YT8M_RGB(torch.utils.data.Dataset):
    def __init__(self, root, clip_length):
        self.root = root
        self.clip_length = clip_length
        self.buffer = clip_length
        self.clips = []
        f = open("durations.json","r")
        self.durations = json.load(f)
        f.close()
        for k, v in self.durations.items():
            for i in range(v//(self.clip_length+self.buffer)):
                self.clips.append((k, i*(self.clip_length+self.buffer)))
        random.seed(42)
        random.shuffle(self.clips)
        

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        vid, l = self.clips[index]
        v, a, m = torchvision.io.read_video(self.root+vid+'.mp4', l//30, (l+self.clip_length)//30, pts_unit = 'sec')
        return v[:self.clip_length, :, :, :]


class YT8M_Flow(torch.utils.data.Dataset):
    def __init__(self, root, clip_length):
        self.root = root
        self.clip_length = clip_length
        self.buffer = clip_length
        self.clips = []
        f = open("durations.json","r")
        self.durations = json.load(f)
        f.close()
        for k, v in self.durations.items():
            for i in range(v//(self.clip_length+self.buffer)):
                self.clips.append((k, i*(self.clip_length+self.buffer)))
        random.seed(42)
        random.shuffle(self.clips)

    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index):
        vid, l = self.clips[index]
        arr = np.load(self.root+vid+'.npy')
        return arr[l//5:(l+self.clip_length)//5, :, :, :]