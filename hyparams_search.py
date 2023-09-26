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

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.autograd.set_detect_anomaly(True)

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

""" 
python hyparams_search.py \
    --config configs/mb/MB_ft_NTU60_xsub.yaml \
    --pretrained /home/osabdelfattah/TCL/mb_pretrained/mb_pretrained_light.bin \
    --checkpoint /home/osabdelfattah/TCL/checkpoint/hyparams_search
    
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
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--of', default=False, action="store_true", help='add temporal pooling')
    opts = parser.parse_args()
    return opts

def load_checkpoint (model, checkpoint, clean=False):
    #checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)

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


def simclr_loss(output_fast,output_slow,normalize=True, temperature_param=0.5):
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

def validate(test_loader, model, criterion, contrastive_loss, log_file_name):
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
        for idx, (batch_input, batch_gt) in (enumerate(test_loader, 0)):
            
            batch_input_pure, batch_input_noisy = batch_input

            batch_size = len(batch_input_pure)    
            #if torch.cuda.is_available():
            batch_gt = batch_gt.cuda()
            batch_input_pure = batch_input_pure.cuda()
            batch_input_noisy = batch_input_noisy.cuda()

            outputs, features_sc, features_cc, j_importances, branch_inps = model(batch_input_pure, batch_input_noisy)
            output, j_importance, branch_inp = outputs[0], j_importances[0], branch_inps[0]

            ce_loss = criterion(output, batch_gt)

            ## contrastive losses for masked branch
            my_features_cc = torch.cat([features_cc[0].unsqueeze(1), features_cc[1].unsqueeze(1)], dim=1)
            class_loss_masked = contrastive_loss(my_features_cc, batch_gt)
            sample_loss_masked = simclr_loss(features_sc[0],features_sc[1])  # From SimClr

            ## contrastive losses for noisy branch
            my_features_cc = torch.cat([features_cc[0].unsqueeze(1), features_cc[2].unsqueeze(1)], dim=1)
            class_loss_noisy = contrastive_loss(my_features_cc, batch_gt)
            sample_loss_noisy = simclr_loss(features_sc[0],features_sc[2]) # From SimClr

            cc_loss = (class_loss_masked + class_loss_noisy) #/ 2 # class contrastive loss
            sc_loss = ((sample_loss_masked + sample_loss_noisy) / 2)# * 9 # sample contrastive loss
            
            total_loss = ce_loss + cc_loss + sc_loss

            # update metric
            total_losses.update(total_loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            sc_losses.update(sc_loss.item(), batch_size)
            cc_losses.update(cc_loss.item(), batch_size)

            acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
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


def train_me(config, opts=None):

    args = config
    print(args)

    log_file_name = opts.checkpoint + "/logs2"+ ".txt"

    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args["finetune"]:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = opts.pretrained#os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args["partial_train"]:
        model_backbone = partial_train_layers(model_backbone, args["partial_train"])
    #model = ActionNet(backbone=model_backbone, dim_rep=args["dim_rep"], num_classes=args["action_classes"], dropout_ratio=args["dropout_ratio"], version=args["model_version"], hidden_dim=args["hidden_dim"], num_joints=args["num_joints"])
    
    model = MaskCLR(backbone=model_backbone,dim_rep=args["dim_rep"], num_classes=args["action_classes"],\
                        dropout_ratio=args["dropout_ratio"], version=args["model_version"], hidden_dim=args["hidden_dim"],\
                            num_joints=args["num_joints"], arch=args["head_version"])
    
    criterion = torch.nn.CrossEntropyLoss()

    contrastive_loss = SupConLoss(temperature=0.07)
    contrastive_loss = contrastive_loss.cuda()

    #if torch.cuda.is_available():
    
    criterion = criterion.cuda() 

    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args["batch_size"],
          'shuffle': True,
          'num_workers': 1,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args["batch_size"],
          'shuffle': False,
          'num_workers': 1,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = '/home/osabdelfattah/TCL/datasets/ntu60/Frames/%s.pkl' % args["dataset"]

    if not opts.evaluate:
        ntu60_xsub_train = NTURGBD(data_path=data_path, data_split=args["data_split"]+'_train', n_frames=args["clip_len"], \
                                   random_move=args["random_move"], scale_range=args["scale_range_train"], of=opts.of)
        train_loader = DataLoader(ntu60_xsub_train, **trainloader_params, drop_last=True)
    
        ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args["data_split"]+'_val', n_frames=args["clip_len"], \
                                 random_move=False, scale_range=args["scale_range_test"])
        test_loader = DataLoader(ntu60_xsub_val, **testloader_params, drop_last=True)


    checkpoint = session.get_checkpoint()


    if opts.resume or opts.evaluate or checkpoint:
        checkpoint_state = checkpoint.to_dict()
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        #checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        #model.load_state_dict(checkpoint['model'], strict=True)

        model, checkpoint_state = load_checkpoint(model, checkpoint_state, clean=True)
        #for i in range(3):
        #    model.module.model_branches[i], _ = load_checkpoint(model.module.model_branches[i], chk_filename, clean=True)
    
    model = nn.DataParallel(model)
    model = model.cuda()

    if not opts.evaluate:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("I will save the results in %s ....." %(log_file_name))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.model.backbone.parameters()), "lr": args["lr_backbone"]},
                  {"params": filter(lambda p: p.requires_grad, model.module.model.head.parameters()), "lr": args["lr_head"]},
            ],      lr=args["lr_backbone"], 
                    weight_decay=args["weight_decay"]
        )

        scheduler = StepLR(optimizer, step_size=1, gamma=args["lr_decay"])
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))

        
        if opts.resume  or checkpoint:
            
            st = checkpoint_state['epoch']
            if 'optimizer' in checkpoint_state and checkpoint_state['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint_state['optimizer'])
            else:
                print('WARNING: this checkpoint_state does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint_state['lr']
            if 'best_acc' in checkpoint_state and checkpoint_state['best_acc'] is not None:
                best_acc = checkpoint_state['best_acc']
        # Training
        
    

        for epoch in range(st, 10):
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


            for idx, (batch_input, batch_gt) in (enumerate(train_loader, 0)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)

                batch_input_pure, batch_input_noisy = batch_input

                batch_size = len(batch_input_pure)
                #if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input_pure = batch_input_pure.cuda()
                batch_input_noisy = batch_input_noisy.cuda()


                #output = model(batch_input_pure) # (N, num_classes)

                outputs, features_sc, features_cc, j_importances, branch_inps = model(batch_input_pure, batch_input_noisy)
                output, j_importance, branch_inp = outputs[0], j_importances[0], branch_inps[0]

                optimizer.zero_grad()
                ce_loss = criterion(output, batch_gt)

                my_features_cc = torch.cat([features_cc[0].unsqueeze(1), features_cc[1].unsqueeze(1)], dim=1)
                #print(my_features_sc.shape,  batch_gt.shape)
                #print(torch.unique(my_features_sc))

                class_loss_masked = contrastive_loss(my_features_cc, batch_gt)
                #sample_loss_masked = contrastive_loss(my_features_sc)  # From SupCont
                sample_loss_masked = simclr_loss(features_sc[0],features_sc[1])  # From SimClr

                #print("sample_loss_masked: ", sample_loss_masked)
                #print("class_loss_masked: ", class_loss_masked)

                my_features_cc = torch.cat([features_cc[0].unsqueeze(1), features_cc[2].unsqueeze(1)], dim=1)
                class_loss_noisy = contrastive_loss(my_features_cc, batch_gt)
                #sample_loss_noisy = contrastive_loss(my_features_sc) # From SupCont
                sample_loss_noisy = simclr_loss(features_sc[0],features_sc[2]) # From SimClr
                #print("sample_loss_noisy: ", sample_loss_noisy)
                #print("class_loss_noisy: ", class_loss_noisy)

                cc_loss = (class_loss_masked + class_loss_noisy) / 2 # class contrastive loss
                sc_loss = ((sample_loss_masked + sample_loss_noisy) / 2)# * 9 # sample contrastive loss

                #print("ce_loss: ", ce_loss )
                #print("sc_loss: ", sc_loss )
                #print("cc_loss: ", cc_loss )
                
                total_loss = ce_loss + args["ccl"]*cc_loss + args["scl"]*sc_loss

                #print(total_loss)
                total_losses.update(total_loss.item(), batch_size)
                ce_losses.update(ce_loss.item(), batch_size)
                sc_losses.update(sc_loss.item(), batch_size)
                cc_losses.update(cc_loss.item(), batch_size)


                acc1, acc5 = accuracy(output, batch_gt, topk=(1, 5))
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
            test_loss, test_top1, test_top5 = validate(test_loader, model, criterion, contrastive_loss, log_file_name)

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
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')

            log = str('Saving checkpoint to' + chk_path)
            log_stuff(log_file_name, log)

            

            

            # Save best checkpoint.
            log = str("Prev. Best Test Top-1: " + str( best_acc))
            log_stuff(log_file_name, log)
            
            if test_top1 > best_acc:
                best_acc = test_top1

                log = "save best checkpoint"
                log_stuff(log_file_name, log)

                acc_name = ""
                if best_acc.item() > 92:
                    acc_name = '_' + str(best_acc.item())

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

    checkpoint_data = {
        'epoch': epoch+1,
        'lr': scheduler.get_last_lr(),
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'best_acc' : best_acc
    }

    torch.save(checkpoint_data, chk_path)

    checkpoint = Checkpoint.from_dict(checkpoint_data)
    session.report(
        {"loss": test_loss, "accuracy": test_top1},
        checkpoint=checkpoint,
    )
            #break

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)

    """
    config = {
        "l1": tune.choice([2 ** i for i in range(9)]),
        "l2": tune.choice([2 ** i for i in range(9)]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }

    lr_backbone: 0.0001
    lr_head: 0.001
    weight_decay: 0.01
    lr_decay: 0.99

    hidden_dim: 2048
    dropout_ratio: 0.5
    head_version: base
    
    """

    args['lr_backbone'] = tune.loguniform(1e-5, 1e-2)
    args['lr_head'] = tune.loguniform(1e-4, 1e-2)

    args['hidden_dim'] = tune.choice([512,1024,2048])
    args['depth'] = tune.choice([2,3,4,5])

    #args['dim_feat'] = tune.choice([256,512])
    args['batch_size'] = tune.choice([4,8])

    args['ccl'] = tune.choice([0.25,0.5,1,2])
    args['scl'] = tune.choice([0.25,0.5,1,2])


    args['head_version'] = tune.choice(["base", 'series', "parallel"])
    
    gpus_per_trial = 2
    max_num_epochs=10
    num_samples=20

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    # ...
    result = tune.run(
        partial(train_me, opts=opts),
        resources_per_trial={"cpu": 30, "gpu": gpus_per_trial},
        config=args,
        num_samples=num_samples,
        scheduler=scheduler)

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    

    """  model_backbone = load_backbone(args)

    
    best_trained_model = MaskCLR(backbone=model_backbone,dim_rep=best_trial.config.dim_rep, num_classes=best_trial.config.action_classes,\
                        dropout_ratio=best_trial.config.dropout_ratio, version=best_trial.config.model_version, hidden_dim=best_trial.config.hidden_dim,\
                            num_joints=best_trial.config.num_joints, arch=best_trial.config.head_version)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc)) """

    #train_me(args, opts)