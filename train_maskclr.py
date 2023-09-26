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

from loss import SupConLoss

import cv2
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle

from matplotlib import pyplot as plt


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.autograd.set_detect_anomaly(True)

""" 
CUDA_VISIBLE_DEVICES=0 python train_maskclr.py.py \
    --config configs/mb/MB_ft_NTU60_xsub.yaml \
    --resume checkpoint/best_epoch_xsub_org.bin \
    --checkpoint checkpoint/resume \
    --print_freq 1 \
    --msk_path_start_epoch 300


python train_maskclr.py.py \
    --config configs/mb/MB_ft_NTU60_xsub.yaml \
    --resume checkpoint/best94.bin \
    --checkpoint checkpoint/pifpaf_of \
    --print_freq 200

python train_maskclr.py.py \
    --config configs/mb/MB_ft_NTU60_xsub.yaml \
    --resume checkpoint/pifpaf_of/latest_epoch.bin \
    --checkpoint checkpoint/pifpaf_of \
    --print_freq 200 \
    
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
    parser.add_argument('-freq', '--print_freq', default=1)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--num_workers', default=1)
    parser.add_argument('--of', default=False, action="store_true", help='add temporal pooling')
    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('--mask_th', default=None, type=float)
    parser.add_argument('--msk_path_start_epoch', default=300, type=float)
    opts = parser.parse_args()
    return opts

def load_checkpoint (model, chk_filename, clean=False):
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)

    state_dict =checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    if clean:
        for k, v in state_dict.items():
            if 'module'  in k:
                k = k.replace('module.', '', 1)
            new_state_dict[k]=v
    
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict, strict=True)

    return model, checkpoint

def sample_contrastive_loss(output_fast,output_slow,normalize=True, temperature_param=0.5):
    out = torch.cat((output_fast, output_slow), dim=0)
    sim_mat = torch.mm(out, torch.transpose(out,0,1))
    if normalize:
        sim_mat_denom = torch.mm(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(1).t())
        sim_mat = sim_mat / sim_mat_denom.clamp(min=1e-16)
    sim_mat = torch.exp(sim_mat / temperature_param)
    if normalize:
        sim_mat_denom = torch.norm(output_fast, dim=1) * torch.norm(output_slow, dim=1)
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / sim_mat_denom / temperature_param)
    else:
        sim_match = torch.exp(torch.sum(output_fast * output_slow, dim=-1) / temperature_param)
    sim_match = torch.cat((sim_match, sim_match), dim=0)
    norm_sum = torch.exp(torch.ones(out.size(0)) / temperature_param )
    norm_sum = norm_sum.cuda()
    loss = torch.mean(-torch.log(sim_match / (torch.sum(sim_mat, dim=-1) - norm_sum)))
    return loss

def validate(test_loader, model, criterion, log_file_name, class_contrastive_loss, epoch):
    model.eval()
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    ce_losses = AverageMeter()
    sc_losses = AverageMeter()
    cc_losses = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()


    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            
            batch_input_pure, batch_input_noisy = batch_input

            batch_size = len(batch_input_pure)    
            #if torch.cuda.is_available():
            batch_gt = batch_gt.cuda()
            batch_input_pure = batch_input_pure.cuda()
            batch_input_noisy = batch_input_noisy.cuda()

            outputs, features_sc, features_cc, j_importances, branch_inps = model(batch_input_pure, batch_input_noisy)
            #output, j_importance, branch_inp = outputs[0], j_importances[0], branch_inps[0]

            sc_loss = cc_loss = ce_loss = 0

            if epoch >= opts.msk_path_start_epoch:

                ce_loss = (criterion(outputs[0], batch_gt) + criterion(outputs[1], batch_gt))/2
                #standard_class_rep = get_class(features_cc[0])
                #masked_class_rep = get_class(features_cc[1])

                masked_class_rep = torch.cat([features_cc[0].unsqueeze(1), features_cc[1].unsqueeze(1)], dim=1)

                class_loss_masked = class_contrastive_loss(masked_class_rep, batch_gt)
                #sample_loss_masked = contrastive_loss(my_features_sc)  # From SupCont
                sample_loss_masked = sample_contrastive_loss(features_sc[0],features_sc[1]) 

                #noisy_class_rep = get_class(features_cc[2])
                #class_loss_noisy = class_contrastive_loss(standard_class_rep, noisy_class_rep)
                #sample_loss_noisy = contrastive_loss(my_features_sc) # From SupCont
                #sample_loss_noisy = sample_contrastive_loss(features_sc[0],features_sc[2]) 
            

                #cc_loss = (class_loss_masked + class_loss_noisy) / 2 # class contrastive loss
                #sc_loss = ((sample_loss_masked + sample_loss_noisy)) / 2 # sample contrastive loss

                sc_loss = sample_loss_masked
                cc_loss = class_loss_masked

                sc_losses.update(sc_loss.item(), batch_size)
                cc_losses.update(cc_loss.item(), batch_size)
                #print("ce_loss: ", ce_loss )
                #print("sc_loss: ", sc_loss )
                #print("cc_loss: ", cc_loss )
            
            else:
                ce_loss = criterion(outputs[0], batch_gt)

            total_loss = ce_loss + args.ccl*cc_loss + args.scl*sc_loss

            #print(total_loss)
            total_losses.update(total_loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)

            acc1, acc5 = accuracy(outputs[0], batch_gt, topk=(1, 5))
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % int(opts.print_freq) == 0:
                log = str('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'total_Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                      'ce_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'sc_Loss {sc_loss.val:.4f} ({sc_loss.avg:.4f})\t'
                      'cc_Loss {cc_loss.val:.4f} ({cc_loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                       idx, len(test_loader), batch_time=batch_time, total_loss = total_losses,
                       ce_loss=ce_losses, sc_loss=sc_losses, cc_loss=cc_losses, \
                        top1=top1, top5=top5))
                
                log_stuff(log_file_name, log)
                sys.stdout.flush()
            
            #break
    return total_losses.avg, top1.avg, top5.avg

def vis_latent_space(all_features, all_targets, file_name):
    embeddings_dic = {}

    plt.figure(figsize=(20, 16))
    embeddings = TSNE(n_jobs=8).fit_transform(all_features)  #(M,T,J,C)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    embeddings_dic['x'] = vis_x
    embeddings_dic['y'] = vis_x
    embeddings_dic['targets'] = all_targets

    with open(file_name + '.pickle', 'wb') as handle:
        pickle.dump(embeddings_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.scatter(vis_x, vis_y, c=all_targets, cmap=plt.cm.get_cmap("jet", 60), marker='.')

    plt.colorbar(ticks=range(60))
    plt.clim(0,60)
    #save_path = 'vis_all.png'
    plt.savefig(file_name + '.png', bbox_inches="tight")

def train_with_config(args, opts):
    print(args)

    log_file_name = opts.checkpoint + "/logs"+ ".txt"

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
            chk_filename = opts.pretrained#os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    #model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    
    model = MaskCLR(backbone=model_backbone,dim_rep=args.dim_rep, num_classes=args.action_classes,\
                        dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim,\
                            num_joints=args.num_joints, arch=args.head_version, mask_th=opts.mask_th)
    
    criterion = torch.nn.CrossEntropyLoss()

    class_contrastive_loss = SupConLoss(temperature=0.07)
    class_contrastive_loss = class_contrastive_loss.cuda()

    #if torch.cuda.is_available():
    
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
          'num_workers': opts.num_workers,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': opts.num_workers,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = '/home/os/contrastive_action_recog/datasets/ntu60/Frames/%s.pkl' % args.dataset

    if not opts.evaluate:
        ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args.data_split+'_train', n_frames=args.clip_len, \
                                   random_move=args.random_move, scale_range=args.scale_range_train, of=opts.of, chunk=opts.chunk)
        train_loader = DataLoader(ntu60_xsub_train, **trainloader_params, drop_last=True)
    
        ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, \
                                 random_move=False, scale_range=args.scale_range_test, chunk=opts.chunk)
        test_loader = DataLoader(ntu60_xsub_val, **testloader_params, drop_last=True)


    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        #checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        #model.load_state_dict(checkpoint['model'], strict=True)

        model.model, checkpoint = load_checkpoint(model.model, chk_filename, clean=True)
        #for i in range(3):
        #    model.module.model_branches[i], _ = load_checkpoint(model.module.model_branches[i], chk_filename, clean=True)
    
    model = nn.DataParallel(model)
    model = model.cuda()

    if not opts.evaluate:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("I will save the results in %s ....." %(log_file_name))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

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
            if 'epoch' in checkpoint:
                st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            
            if 'lr' in checkpoint:
                lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
        # Training
        
    

        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            total_losses = AverageMeter()
            ce_losses = AverageMeter()
            sc_losses = AverageMeter()
            cc_losses = AverageMeter()

            top1 = AverageMeter()
            top5 = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)


            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)

                batch_input_pure, batch_input_noisy = batch_input

                #print(batch_input_pure.shape)
                batch_size = len(batch_input_pure)
                #if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input_pure = batch_input_pure.cuda()
                batch_input_noisy = batch_input_noisy.cuda()


                #output = model(batch_input_pure) # (N, num_classes)

                outputs, features_sc, features_cc, j_importances, branch_inps = model(batch_input_pure, batch_input_noisy)
                #output, j_importance, branch_inp = outputs[0], j_importances[0], branch_inps[0]

                optimizer.zero_grad()

                #output_ce = (outputs[0] + outputs[1] + outputs[2]) / 3

                
                sc_loss = cc_loss = ce_loss = 0

                if epoch >= opts.msk_path_start_epoch:

                    ce_loss = (criterion(outputs[0], batch_gt) + criterion(outputs[1], batch_gt))/2

                    masked_class_rep = torch.cat([features_cc[0].unsqueeze(1), features_cc[1].unsqueeze(1)], dim=1)
                    class_loss_masked = class_contrastive_loss(masked_class_rep, batch_gt)
                    #sample_loss_masked = contrastive_loss(my_features_sc)  # From SupCont
                    sample_loss_masked = sample_contrastive_loss(features_sc[0],features_sc[1]) 

                    #noisy_class_rep = get_class(features_cc[2])
                    #class_loss_noisy = class_contrastive_loss(standard_class_rep, noisy_class_rep)
                    #sample_loss_noisy = contrastive_loss(my_features_sc) # From SupCont
                    #sample_loss_noisy = sample_contrastive_loss(features_sc[0],features_sc[2]) 
                

                    #cc_loss = (class_loss_masked + class_loss_noisy) / 2 # class contrastive loss
                    #sc_loss = ((sample_loss_masked + sample_loss_noisy)) / 2 # sample contrastive loss

                    sc_loss = sample_loss_masked
                    cc_loss = class_loss_masked

                    sc_losses.update(sc_loss.item(), batch_size)
                    cc_losses.update(cc_loss.item(), batch_size)
                    #print("ce_loss: ", ce_loss )
                    #print("sc_loss: ", sc_loss )
                    #print("cc_loss: ", cc_loss )
                
                else:
                    ce_loss = criterion(outputs[0], batch_gt)

                total_loss = ce_loss + args.ccl*cc_loss + args.scl*sc_loss

                #print(total_loss)
                total_losses.update(total_loss.item(), batch_size)
                ce_losses.update(ce_loss.item(), batch_size)
                


                acc1, acc5 = accuracy(outputs[0], batch_gt, topk=(1, 5))
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                total_loss.backward()

                optimizer.step()    
                batch_time.update(time.time() - end)
                end = time.time()
                if (idx + 1) % int(opts.print_freq) == 0:
                    log = str('Train: [{0}][{1}/{2}]\t'
                        'total_loss {total_loss.val:.3f} ({total_loss.avg:.3f})\t'
                        'ce_loss {ce_loss.val:.3f} ({ce_loss.avg:.3f})\t'
                        'sc_loss {sc_loss.val:.3f} ({sc_loss.avg:.3f})\t'
                        'cc_loss {cc_loss.val:.3f} ({cc_loss.avg:.3f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Best_Acc@1 {best_acc:.3f} ({best_acc:.3f})'.format(
                        epoch, idx + 1, len(train_loader), batch_time=batch_time, total_loss = total_losses,
                        data_time=data_time, ce_loss=ce_losses, sc_loss=sc_losses,\
                            cc_loss=cc_losses,top1=top1, best_acc=best_acc))
                
                    log_stuff(log_file_name, log)
                    sys.stdout.flush()

                #break
            #break    
            test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, log_file_name, class_contrastive_loss, epoch)


            log = ('Overall test Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))
            
            log_stuff(log_file_name, log)
            sys.stdout.flush()


            train_writer.add_scalar('train_total_loss', total_losses.avg, epoch + 1)    
            train_writer.add_scalar('train_ce_loss', ce_losses.avg, epoch + 1)
            train_writer.add_scalar('train_sc_loss', sc_losses.avg, epoch + 1)
            train_writer.add_scalar('train_cc_loss', cc_losses.avg, epoch + 1)
            train_writer.add_scalar('train_top1', top1.avg, epoch + 1)
            train_writer.add_scalar('train_top5', top5.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            train_writer.add_scalar('test_top5', test_top5, epoch + 1)
            
            scheduler.step()

            # Save latest checkpoint.

            acc_name = ""
            if test_top1 > 90:
                acc_name = '_' + str(round(test_top1,2))

            chk_path = os.path.join(opts.checkpoint, 'latest_epoch%s.bin' % (acc_name).format(epoch))

            #chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')

            log = str('Saving checkpoint to' + chk_path)
            log_stuff(log_file_name, log)

            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            log = str("Prev. Best Test Top-1: " + str( best_acc))
            log_stuff(log_file_name, log)
            
            if test_top1 > best_acc:
                best_acc = test_top1

                log = "save best checkpoint"
                log_stuff(log_file_name, log)

                acc_name = ""
                if best_acc.item() > 94:
                    acc_name = '_' + str(round(best_acc.item(),2))

                best_chk_path = os.path.join(opts.checkpoint, 'best_epoch%s.bin' % (acc_name).format(epoch))

                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)


            log = str("Now Best Test Top-1: "+ str( best_acc))
            log_stuff(log_file_name, log)
            #break


    if opts.evaluate:
        test_loss, test_top1, test_top5 = validate(test_loader, model, criterion)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'
              'Acc@5 {top5:.3f} \t'.format(loss=test_loss, top1=test_top1, top5=test_top5))

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)