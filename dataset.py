import cv2
import torchvision
import numpy as np
import torch
import os
import json
import random
from torchvision import transforms
import moviepy.editor as mp
from PIL import Image


class YT8M_RGB(torch.utils.data.Dataset):
    def __init__(self, transform, root, clip_length):
        self.root = root
        self.clip_length = clip_length
        self.transform = transform
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
        
        v = []
        cap = cv2.VideoCapture(self.root+vid+'.mp4')
        cap.set(1, l)
        for i in range(self.clip_length):
            _, f = cap.read()
            v.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        v = v*2
        return torch.stack(self.transform(v), 1)
        
#         clip = mp.VideoFileClip(self.root+vid+'.mp4')
#         v = [Image.fromarray(np.uint8(clip.get_frame(i))).convert('RGB') for i in range(l, l+self.clip_length)]
#         print(len(v))
        
#         v, a, m = torchvision.io.read_video(self.root+vid+'.mp4', l/30, (l+self.clip_length*2)/30, pts_unit='sec')
#         v = v.permute(0, 3, 1, 2)
#         assert v.shape[0]>=self.clip_length, str(v.shape)+" "+str(l)+" "+str(self.durations[vid])
#         v = [transforms.ToPILImage()(v[i]).convert("RGB") for i in range(self.clip_length)]
#         v = v*2
#         return torch.stack(self.transform(v), 1)


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