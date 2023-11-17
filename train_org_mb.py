import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

"""
python train_org_mb.py \
    --config configs/action/MB_train_NTU60_xsub.yaml \
    --checkpoint checkpoint/of-60sub-maskclrv2 \
    --resume checkpoint/of-60sub-maskclrv2/latest_epoch.bin \
    --print_freq 100

python train_org_mb.py \
    --config configs/action/MB_train_NTU60_xsub.yaml \
    --checkpoint checkpoint/of-60sub-mb-org \
    --pretrained /home/osabdelfattah/TCL/mb_pretrained/mb_pretrained_light.bin \
    --print_freq 100

"""

def log_stuff(log_file, log):
    print (log)
    #if (not is_evaluate):
    with open(log_file, "a") as myfile:
        myfile.write(log + "\n")
        myfile.flush()
    myfile.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', type=int,default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    
    parser.add_argument('--of', default=False, action="store_true", help='add temporal pooling')
    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--mask_th', default=None, type=float)
    parser.add_argument('--msk_path_start_epoch', default=300, type=float)
    parser.add_argument('--msk_type', default='', type=str)
    parser.add_argument('--cl_type', default='tcl', type=str)
    parser.add_argument('--not_strict', default=True, action="store_false")
    opts = parser.parse_args()
    return opts

def validate(test_loader, model, criterion, log_file_name):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in (enumerate(test_loader)):
            batch_input = batch_input
            batch_size = len(batch_input)    
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output, _,_,_ = model(batch_input)    # (N, num_classes)
            #output = output[0]
            loss = criterion(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                log = str('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, top1=top1, top5=top5))

                log_stuff(log_file_name, log)
                sys.stdout.flush()
    return losses.avg, top1.avg, top5.avg


def train_with_config(args, opts):
    log_file_name = opts.checkpoint + "/logs"+ ".txt"
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = opts.pretrained #os.path.join(opts.pretrained, opts.selection)
            log = str('Loading backbone ' + chk_filename)
            log_stuff(log_file_name, log)
            sys.stdout.flush()
            
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    # model = MaskCLRv2(backbone=model_backbone,dim_rep=args.dim_rep, num_classes=args.action_classes,\
    #                         dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim,\
    #                             num_joints=args.num_joints, arch=args.head_version, mask_th=opts.mask_th)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda() 
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'drop_last': True
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True,
          'drop_last': True
    }
    data_path = '/home/osabdelfattah/MaskCLR/datasets/ntu60/%s.pkl' % args.dataset

    ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, \
                                random_move=args.random_move, scale_range=args.scale_range_train, of=opts.of, chunk=opts.chunk)
    train_loader = DataLoader(ntu60_xsub_train, **trainloader_params)

    ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, \
                                random_move=False, scale_range=args.scale_range_test, chunk=opts.chunk)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params)
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        log = str('Loading checkpoint ' + chk_filename)
        log = log_stuff(log_file_name, log)
        sys.stdout.flush()
        
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    
    if not opts.evaluate:
        # optimizer = optim.AdamW(
        #     [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
        #           {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
        #     ],      lr=args.lr_backbone, 
        #             weight_decay=args.weight_decay
        # )

        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.model.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.module.model.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt) in (enumerate(train_loader)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)

                batch_input = batch_input

                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                output, _,_,_ = model(batch_input) # (N, num_classes)
                #output = output[0]
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                loss_train.backward()
                optimizer.step()    
                batch_time.update(time.time() - end)
                end = time.time()
                if (idx + 1) % opts.print_freq == 0:
                    log = str('Train: [{0}][{1}/{2}]\t'
                        'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses_train, top1=top1))
                    log_stuff(log_file_name, log)
                    sys.stdout.flush()
                
            test_loss, test_top1, test_top5 = validate(test_loader, model, criterion,log_file_name)

            log = ('Overall test Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))
            
            log_stuff(log_file_name, log)

            sys.stdout.flush()
                
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('test_top5', test_top5, epoch + 1)
            
            scheduler.step()

            # Save latest checkpoint.
            acc_name = ""
            if test_top1 > 90:
                acc_name = '_' + str(round(test_top1.item(),2))

            chk_path = os.path.join(opts.checkpoint, 'latest_epoch%s.bin' % (acc_name).format(epoch))

            log = str('Saving checkpoint to' + chk_path)
            log_stuff(log_file_name, log)
            sys.stdout.flush()

            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

            log = str("Prev. Best Test Top-1: " + str( best_acc))
            log_stuff(log_file_name, log)

            if test_top1 > best_acc:
                best_acc = test_top1
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)
            
            log = str("Now Best Test Top-1: "+ str( best_acc))
            log_stuff(log_file_name, log)

    if opts.evaluate:
        test_loss, test_top1, test_top5 = validate(test_loader, model, criterion,log_file_name)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)