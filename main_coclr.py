import os
import sys
import argparse
import time, re
import builtins
import numpy as np
import random 
import pickle 
import math 
from tqdm import tqdm 
from backbone.select_backbone import select_backbone

import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils import data 
from torchvision import transforms
import torchvision.utils as vutils

import utils.augmentation as A
import utils.transforms as T
import utils.tensorboard_utils as TB
from tensorboardX import SummaryWriter

from utils.utils import AverageMeter, write_log, calc_topk_accuracy, calc_mask_accuracy, \
batch_denorm, ProgressMeter, neq_load_customized, save_checkpoint, Logger, FastDataLoader
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn
from model.pretrain import CoCLR
from dataset import YT8M_RGB, YT8M_Flow, YT8M_RGB_Flow

def main(args):
    model = CoCLR(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t, topk=args.topk, reverse=args.reverse)
    if args.reverse:
        print('[Warning] using RGB-Mining to help flow')
    else:
        print('[Warning] using Flow-Mining to help RGB')
    args.num_seq = 2
        
    args.img_path, args.model_path, args.exp_path = set_path(args)
    args.writer_train = SummaryWriter(logdir='runs')

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    args.iteration = 1

    ### data ###  
    transform_train = get_transform('train', args)
    train_loader = get_dataloader(get_data(transform_train, args), args)
    transform_train_cuda = transforms.Compose([
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225], channel=1)])
    n_data = len(train_loader.dataset)

    ### restart training ### 
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']+1
            args.iteration = checkpoint['iteration']
            best_acc = checkpoint['best_acc']
            state_dict = checkpoint['state_dict']

            try: model_without_ddp.load_state_dict(state_dict)
            except: 
                print('[WARNING] Non-Equal load for resuming training!')
                neq_load_customized(model_without_ddp, state_dict, verbose=True)

            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer'])

    elif args.pretrain != ['random', 'random']:
        # first path: weights to be trained
        # second path: weights as the oracle, not trained
        if os.path.isfile(args.pretrain[1]): # second network --> load as sampler
            checkpoint = torch.load(args.pretrain[1], map_location=torch.device('cpu'))
            second_dict = checkpoint['state_dict']
            new_dict = {}
            for k,v in second_dict.items(): # only take the encoder_q
                if 'encoder_q.' in k:
                    k = k.replace('encoder_q.', 'sampler.')
                    new_dict[k] = v
            second_dict = new_dict

            new_dict = {} # remove queue, queue_ptr
            for k, v in second_dict.items():
                if 'queue' not in k:
                    new_dict[k] = v 
            second_dict = new_dict
        if os.path.isfile(args.pretrain[0]): # first network --> load both encoder q & k
            checkpoint = torch.load(args.pretrain[0], map_location=torch.device('cpu'))
            first_dict = checkpoint['state_dict']

            new_dict = {} # remove queue, queue_ptr
            for k, v in first_dict.items():
                if 'queue' not in k:
                    new_dict[k] = v 
            first_dict = new_dict

            # update both q and k with q
            new_dict = {}
            for k,v in first_dict.items(): # only take the encoder_q
                if 'encoder_q.' in k:
                    new_dict[k] = v
                    k = k.replace('encoder_q.', 'encoder_k.')
                    new_dict[k] = v
            first_dict = new_dict

        state_dict = {**first_dict, **second_dict}
        try:
            del state_dict['queue_label'] # always re-fill the queue
        except:
            pass 
        neq_load_customized(model_without_ddp, state_dict, verbose=True)
    torch.backends.cudnn.benchmark = True
    best_acc = 0

    ### main loop ###    
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        _, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, transform_train_cuda, epoch, args)
        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):         
            is_best = train_acc > best_acc
            best_acc = max(train_acc, best_acc)
            state_dict = model.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration}
            save_checkpoint(save_dict, is_best, gap=args.save_freq, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=False)
    
    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))

def multi_nce_loss(logits, mask):
    mask_sum = mask.sum(1)
    loss = - torch.log( (F.softmax(logits, dim=1) * mask).sum(1) )
    return loss.mean()

def train_one_epoch(data_loader, model, criterion, optimizer, transforms_cuda, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')
    top1_self_meter = AverageMeter('Self-acc@1', ':.4f')
    top5_self_meter = AverageMeter('Self-acc@5', ':.4f')
    sacc_meter = AverageMeter('Sampling-Acc@%d' % args.topk, ':.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter, top1_self_meter, top5_self_meter, sacc_meter],
        prefix='Epoch:[{}]'.format(epoch))
    model.train() 
    model.sampler.eval() # the sampler is always fixed

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B,3,args.num_seq,args.seq_len,args.img_dim,args.img_dim).transpose(1,2).contiguous()

    for idx, (input_seq, vname) in enumerate(data_loader):
        B = input_seq[0].size(0)

        input_seq = [tr(i.cuda(non_blocking=True)) for i in input_seq]
        vname = vname.cuda(non_blocking=True)

        output, mask = model(*input_seq, vname)
        mask_sum = mask.sum(1)

        loss = multi_nce_loss(output, mask)

        top1, top5 = calc_mask_accuracy(output, mask, (1,5))
        top1_self, top5_self = calc_topk_accuracy(output, torch.zeros(B, dtype=torch.long).cuda(), (1,5))

        del output
        
        losses.update(loss.item(), B)
        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)
        top1_self_meter.update(top1_self.item(), B)
        top5_self_meter.update(top5_self.item(), B)
        
        if model.queue_is_full:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        del loss 
        torch.cuda.empty_cache()
#         if idx % args.print_freq == 0:
#             progress.display(idx)
#             args.writer_train.add_scalar('local/loss', losses.local_avg, args.iteration)
#             args.writer_train.add_scalar('local/top1', top1_meter.local_avg, args.iteration)
#             args.writer_train.add_scalar('local/top5', top5_meter.local_avg, args.iteration)
#             args.writer_train.add_scalar('local/self-top1', top1_self_meter.local_avg, args.iteration)
#             args.writer_train.add_scalar('local/self-top5', top5_self_meter.local_avg, args.iteration)
        args.iteration += 1
    print('(Epoch: [{0}][loss: {1} acc:{2}]\t'.format(epoch,losses.avg, top1_self_meter.avg))
    args.writer_train.add_scalar('global/loss', losses.avg, epoch)
    args.writer_train.add_scalar('global/top1', top1_meter.avg, epoch)
    args.writer_train.add_scalar('global/top5', top5_meter.avg, epoch)
    args.writer_train.add_scalar('global/self-top1', top1_self_meter.avg, epoch)
    args.writer_train.add_scalar('global/self-top5', top5_self_meter.avg, epoch)
    return losses.avg, top1_meter.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    # stepwise lr schedule
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        


def get_transform(mode, args):
    seq_len = args.seq_len * 2 # for both rgb and flow

    null_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=seq_len, bottom_area=0.2),
        A.RandomHorizontalFlip(consistent=False, seq_len=seq_len),
        A.ToTensor(),
    ])

    base_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=seq_len, bottom_area=0.2),
        transforms.RandomApply([
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0, consistent=False, seq_len=seq_len)
            ], p=0.8),
        A.RandomGray(p=0.2, seq_len=seq_len),
        transforms.RandomApply([A.GaussianBlur([.1, 2.], seq_len=seq_len)], p=0.5),
        A.RandomHorizontalFlip(consistent=False, seq_len=seq_len),
        A.ToTensor(),
    ])
    transform = A.TransformController(
                    [A.TwoClipTransform(base_transform, null_transform, seq_len=seq_len, p=0.3),
                     A.OneClipTransform(base_transform, null_transform, seq_len=seq_len)],
                    weights=[0.5,0.5])
    return transform

def get_dataloader(dataset, args):
    data_loader = FastDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    return data_loader

def get_data(transform, args):
    root_rgb = "youtube8m_flow/"
    root_flow = "youtube8m_rgb/"
    dataset = YT8M_RGB_Flow(root_rgb, root_flow, args.seq_len, transform, temp_aug=True)
    return dataset 

def set_path(args):
    if args.resume: exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'log-{args.prefix}/{args.name_prefix}-top{args.topk}{0}_k{args.moco_k}-{args.img_dim}_{args.net}_\
bs{args.batch_size}_lr{args.lr}_seq{args.num_seq}_len{args.seq_len}'.format(
                    '-R' if args.reverse else '', \
                    args=args)
    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    return img_path, model_path, exp_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='s3d', type=str)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--seq_len', default=32, type=int, help='number of frames in each video block')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default=['random', 'random'], nargs=2, type=str, help='path of pretrained model: rgb, flow')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of eval')
    parser.add_argument('--img_dim', default=128, type=int)
    parser.add_argument('--prefix', default='pretrain', type=str)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('-j', '--workers', default=16, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=2048, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--cos', action='store_true', 
                        help='use cosine lr schedule')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)