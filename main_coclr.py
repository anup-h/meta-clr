import argparse
from model.pretrain import InfoNCE, UberNCE
import utils.augmentation as A
from dataset import YT8M_RGB, YT8M_Flow
from torchvision import transforms
from torch.utils import data 
from utils.utils import AverageMeter, write_log, calc_topk_accuracy, calc_mask_accuracy, batch_denorm, ProgressMeter, neq_load_customized, save_checkpoint, Logger, FastDataLoader
import torch.nn as nn
import utils.transforms as T
import torch.optim as optim
from tqdm import tqdm 


def main(args):
    model = InfoNCE(args.net, args.moco_dim, args.moco_k, args.moco_m, args.moco_t)
    transform = get_transform(args)
    ds = YT8M_RGB(transform, 'youtube8m_rgb/', 32)
    
    print(ds[0].shape)
    params = []
    for name, param in model.named_parameters():
        params.append({'params': param})
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    dl = get_dataloader(ds, args)
    transform_train_cuda = transforms.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225], channel=1)])
    train_one_epoch(dl, model, criterion, optimizer, transform_train_cuda, 1, args)
    
def train_one_epoch(data_loader, model, criterion, optimizer, transforms_cuda, epoch, args):

    model.train() 

    def tr(x):
        B = x.size(0)
        return transforms_cuda(x).view(B,3,args.num_seq,args.seq_len,args.img_dim,args.img_dim).transpose(1,2).contiguous()

    for idx, input_seq in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
        B = input_seq.size(0)
        input_seq = tr(input_seq)
        
        output, target = model(input_seq)
        loss = criterion(output, target)
        
        top1, top5 = calc_topk_accuracy(output, target, (1,5))

#         top1_meter.update(top1.item(), B)
#         top5_meter.update(top5.item(), B)
#         losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss.item())

#         batch_time.update(time.time() - end)
#         end = time.time()
        
#         progress.display(idx)

#         if idx % args.print_freq == 0:
#             if args.print:
#                 args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
#                 args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
        
#         args.iteration += 1

#     print('({gpu:1d})Epoch: [{0}][{1}/{2}]\t'
#           'T-epoch:{t:.2f}\t'.format(epoch, idx, len(data_loader), gpu=args.rank, t=time.time()-tic))
    
#     if args.print:
#         args.train_plotter.add_data('global/loss', losses.avg, epoch)
#         args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
#         args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)

#     return losses.avg, top1_meter.avg

def get_dataloader(dataset, args):
    data_loader = FastDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    return data_loader

def get_transform(args):
    null_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=args.seq_len, bottom_area=0.2),
        A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
        A.ToTensor(),
    ])

    base_transform = transforms.Compose([
        A.RandomSizedCrop(size=args.img_dim, consistent=False, seq_len=args.seq_len, bottom_area=0.2),
        transforms.RandomApply([
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0, consistent=False, seq_len=args.seq_len)
            ], p=0.8),
        A.RandomGray(p=0.2, seq_len=args.seq_len),
        transforms.RandomApply([A.GaussianBlur([.1, 2.], seq_len=args.seq_len)], p=0.5),
        A.RandomHorizontalFlip(consistent=False, seq_len=args.seq_len),
        A.ToTensor(),
    ])

    # oneclip: temporally take one clip, random augment twice
    # twoclip: temporally take two clips, random augment for each
    # merge oneclip & twoclip transforms with 50%/50% probability
    transform = A.TransformController(
                    [A.TwoClipTransform(base_transform, null_transform, seq_len=args.seq_len, p=0.3),
                     A.OneClipTransform(base_transform, null_transform, seq_len=args.seq_len)],
                    weights=[0.5,0.5])
    print(transform)
    return transform 
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='s3d', type=str) # r18-all
    parser.add_argument('--model', default='infonce', type=str)
    parser.add_argument('--dataset', default='ucf101-2clip', type=str)
    parser.add_argument('--seq_len', default=32, type=int, help='number of frames in each video block')
    parser.add_argument('--num_seq', default=2, type=int, help='number of video blocks')
    parser.add_argument('--ds', default=1, type=int, help='frame down sampling rate')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--resume', default='', type=str, help='path of model to resume')
    parser.add_argument('--pretrain', default='', type=str, help='path of pretrained model')
    parser.add_argument('--test', default='', type=str, help='path of model to load and pause')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--print_freq', default=5, type=int, help='frequency of printing output during training')
    parser.add_argument('--save_freq', default=1, type=int, help='frequency of eval')
    parser.add_argument('--reset_lr', action='store_true', help='Reset learning rate when resume training?')
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
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
