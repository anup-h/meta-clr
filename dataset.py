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

# This Dataset can return either rgb frames or optical flow, depending on the root that's passed in
class YT8M_Single_Modality(torch.utils.data.Dataset):
    def __init__(self, root, clip_length, transform, temp_aug=True):
        self.root = root
        self.clip_length = clip_length
        self.transform = transform
        self.buffer = 2048
        self.clips = []
        self.temp_aug = temp_aug
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
        vid, start = self.clips[index]
        rgb = np.load(self.root+vid+'.npy')
        start1 = start
        start2 = start
        if self.temp_aug:
            start1 = start1 + np.random.choice(range(self.clip_length))
            start2 = start2 + np.random.choice(range(self.clip_length))
        end1 = start1 + self.clip_length
        end2 = start2 + self.clip_length
        seq_idx = np.concatenate([np.arange(start1,end1), np.arange(start2,end2)])
        seq_rgb = [Image.fromarray(rgb[i,:,:,:].squeeze()) for i in seq_idx]
        return torch.stack(self.transform(seq_rgb), 1)
    
class YT8M_RGB_Flow(torch.utils.data.Dataset):
    def __init__(self, root_rgb, root_flow, clip_length, transform, temp_aug=True):
        self.root_rgb = root_rgb
        self.root_flow = root_flow
        self.clip_length = clip_length
        self.transform = transform
        self.temp_sampling = 128
        self.buffer = 2048
        self.clips = []
        self.temp_aug = temp_aug
        f = open("durations.json","r")
        self.durations = json.load(f)
        f.close()
        instance_number = 0
        for k, v in self.durations.items():
            for i in range(v//(self.clip_length+self.buffer)):
                self.clips.append((k, i*(self.clip_length+self.buffer), instance_number))
                instance_number = instance_number + 1
        random.seed(42)
        random.shuffle(self.clips)

    def __len__(self):
        return len(self.clips)
    
    def frame_sampler(self, start, rgb, flow):
        total_time = rgb.shape[0]
        start1 = start
        start2 = start
        if self.temp_aug:
            start1 = start1 + np.random.choice(range(self.clip_length))
            start2 = start2 + np.random.choice(range(self.clip_length))
        end1 = start1 + self.clip_length
        end2 = start2 + self.clip_length
        seq_idx = np.concatenate([np.arange(start1,end1), np.arange(start2,end2)])
        seq_rgb = [Image.fromarray(rgb[i,:,:,:].squeeze()) for i in seq_idx] # create a list of images for preprocessing
        seq_flow = [Image.fromarray(flow[i,:,:,:].squeeze()) for i in seq_idx]
        return seq_rgb, seq_flow
    
    def __getitem__(self, index):
        vid, l, instance_id = self.clips[index]
        rgb = np.load(self.root_rgb+vid+'.npy')
        flow = np.load(self.root_flow+vid+'.npy')
        seq_rgb, seq_flow = self.frame_sampler(l,rgb,flow)
        transform = self.transform
        seq = transform(seq_rgb[0:self.clip_length] + seq_flow[0:self.clip_length] \
                   + seq_rgb[self.clip_length::] + seq_flow[self.clip_length::])
        seq1 = seq[0:self.clip_length*2] # rgb, flow
        seq2 = seq[self.clip_length*2::] # rgb, flow
        seq1 = torch.stack(seq1, 1)
        seq2 = torch.stack(seq2, 1)
        return (seq1, seq2), instance_id