#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from utils.utils import neq_load_customized
from model.pretrain import InfoNCE
from model.classifier import LinearClassifier
import utils.transforms as T
from utils.utils import calc_topk_accuracy


# In[3]:


class EvalSupervised(pl.LightningModule):
    def __init__(self, unsupervised_path):
        super().__init__()
        self.save_hyperparameters()
        
        self.feat_ext = LinearClassifier(
                    network='s3d', 
                    num_class=101,
                    dropout=0.9,
                    use_dropout=True,
                    use_final_bn=False,
                    use_l2_norm=False)
        checkpoint = torch.load(unsupervised_path)
        state_dict = checkpoint['state_dict']
        new_dict = {}
        for k,v in state_dict.items():
            k = k.replace('encoder_q.0.', 'backbone.')
            new_dict[k] = v
        state_dict = new_dict
        neq_load_customized(self.feat_ext, state_dict, verbose=False)

    def forward(self, x):
        ts = transforms.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1)])
        return self.feat_ext(ts(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = F.cross_entropy(y_hat, y)
        top1, top5 = calc_topk_accuracy(y_hat, y, (1,5))
        tensorboard_logs = {'val_loss': loss, 'top_1': top1, 'top_5': top5}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# In[4]:


m = EvalSupervised('CoCLR-ucf101-rgb-128-s3d-ep182.tar')


# In[ ]:





# In[5]:


def collate(batch):
    lstv = []
    lstl = []
    for v, _, l in batch:
        lstv.append(F.interpolate(torch.as_tensor(v/255.0).permute(3, 0, 1, 2).unsqueeze(0), size=(32, 128, 128), mode='trilinear').squeeze(0))
        lstl.append(l)
    return torch.stack(lstv), torch.as_tensor(lstl)


# In[6]:


ds = torchvision.datasets.UCF101('.//UCF-101', './ucfTrainTestlist', 32, train=True)
# dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)


# In[7]:


vds = torchvision.datasets.UCF101('./UCF-101', './ucfTrainTestlist', 32, train=False)
# vdl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)


# In[8]:


dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)
vdl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate)


# In[ ]:


torch.cuda.empty_cache()
trainer = pl.Trainer(gpus=2, progress_bar_refresh_rate=20, max_epochs=10, accelerator='ddp')
trainer.fit(m, dl, vdl) 


# In[ ]:




