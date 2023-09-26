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

from matplotlib import pyplot as plt
from numpy import interp

import cv2
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.autograd.set_detect_anomaly(True)

""" 
CUDA_VISIBLE_DEVICES=1 python inf_mb.py \
    --config configs/mb/MB_ft_NTU60_xsub.yaml \
    --evaluate checkpoint/org99_h.bin \
    --checkpoint exps/org99_h \
    --num_workers 8 \
    --batch_size 4 \
    --print_freq 200 \
    --chunk 100 \
    --ignore_check

    --evaluate /home/ossamaabdelfattah/MotionBERT/checkpoint/mb_rc_of_val_fixed_t_t003.bin \ ==> for joint activations

    --config /home/ossamaabdelfattah/MotionBERT/configs/action/MB_ft_NTU60_xsub.yaml \
    --evaluate checkpoint/best_epoch_xsub_org.bin \    
    
    --chunk 100 \
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
    parser.add_argument('-freq', '--print_freq', default=200, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--chunk', default=None, type=int)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--of', default=False, action="store_true", help='add temporal pooling')
    parser.add_argument('--ignore_check', default=True, action="store_true", help='add temporal pooling')
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
                k = k.replace('module.', '', 2)
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

def show_skeleton(batch_gt, locations, out,sig, ids, out_dir, j_importances_org = None):

    ntu_categories = ['drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 
        'throw', 'sitting down', 'standing up (from sitting position)', 'clapping', 'reading', 
        'writing', 'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 
        'take off a shoe', 'wear on glasses', 'take off glasses', 'put on a hat/cap', 
        'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 
        'reach into pocket', 'hopping (one foot jumping)', 'jump up', 
        'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard', 
        'pointing to something with finger', 'taking a selfie', 'check time (from watch)', 
        'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute', 
        'put the palms together', 'cross hands in front (say stop)', 'sneeze/cough', 
        'staggering', 'falling', 'touch head (headache)', 'touch chest (stomachache/heart pain)', 
        'touch back (backache)', 'touch neck (neckache)', 'nausea or vomiting condition', 
        'use a fan (with hand or paper)/feeling warm', 'punching/slapping other person', 
        'kicking other person', 'pushing other person', 'pat on back of other person', 
        'point finger at the other person', 'hugging other person', 
        'giving something to other person', "touch other person's pocket", 'handshaking', 
        'walking towards each other', 'walking apart from each other']


    batch_gt = batch_gt.cpu().detach().numpy()

    #print("locations.shape: ", locations.shape)

    locations = locations[:,:,:,:,:2]
    #locations = locations.permute(0,4,2,3,1)
    locations = locations.cpu().detach().numpy()

    out = out.cpu().detach().numpy()

    pure = "_pure" if "pure" in out_dir else ""
    noisy = "_noisy" if "noisy" in out_dir else ""
    robust = "_robust" if "robust" in out_dir else ""
    masked = "_masked" if "masked" in out_dir else ""

    pred = np.argmax(out, 1)#.cpu().detach().numpy()

    #frames = [*range(0, 243, 1)]
    
    sig_print = 0.0 if "pure" in out_dir else sig

    new_printed = 0

    #print(locations.shape)

    for vid_idx in ids:

        pred_class = pred[vid_idx] + 1
        actual_class = batch_gt[vid_idx] + 1

        #if pred_class == actual_class and vis_wrong:
        #    continue
        
        fps = 60 

        location = locations[vid_idx,:,:,:,:]   # N, M, T, J, C
        #print(location.shape)
        #C, T, V, M = location.shape  #M, T, V, C  

        M, T, V, C = location.shape 
        

        #result = np.einsum('kc,ctvm->ktvm', weight, feature) 


        #location = interp(location,[np.min(location),np.max(location)],[0,1360])#(, , , 0, 1360)

        location = (location + 1) / 2

        #print(np.min(location), np.max(location))
        #print(np.unique(location))

        connecting_joint = np.array([8, 1, 2, 3,1,5,6,9,10,11,10,9,12,13,9,15,16])
        #result = result[visualize_class-1,:,:17,:] #25


        if j_importances_org is not None:
            j_importances = j_importances_org.cpu().detach().numpy()
            result = j_importances[vid_idx,:,:,:]  # N, M, T, J
            result = np.maximum(result, 0)
            result = result/np.max(result)

            #print("result: ", result)


        #mm, tt, jj, _ = result.shape
        #result = np.reshape(result, ())

        #print("3: ", result.shape)

        probably_value = out[vid_idx, pred_class-1] #/100
        show_n = 6


        

        #row_sums = result.max(axis=1)
        #result = result / row_sums[:, np.newaxis]


        #if len(frames) > 0:
            #print(location.shape)
            #location = location[:,frames,:,:]
        plt.figure()

        try:

            #print(actual_class)
            action_name = str(ntu_categories[actual_class-1])

            #print(action_name)
            action_name = action_name.replace(' ', "_")
            action_name = action_name.replace('/', "_")
            action_name = action_name.replace('(', "")
            action_name = action_name.replace(')', "")

            #print(out_dir+ action_name)
            os.makedirs(out_dir+ action_name)


            print("writing to %s ... " % str(out_dir+ action_name))

            new_printed+=1
        
        except:
            print("vis class %s already exists. skipping... " % str(out_dir+ action_name))
            continue

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        #print(result.shape, location.shape)

        #print(result.shape[0])
        for t in tqdm(range(T)):
            
            """ if t%2 == 0 and t!= 0:
                continue """

            
            plt.figure(figsize=(8, 6))
            plt.cla()
            #plt.xlim(-50, 2000)
            #plt.ylim(-50, 1100)

            plt.xlim(0, 1)
            plt.ylim(0, 1)
            

            plt.title('frame: {}, \nconfidence: {:.2f}%, \npred_class: {}, \nground_truth: {}, \nsigma: {}'.format(
                        t, probably_value*100, ntu_categories[pred_class-1], \
                            ntu_categories[actual_class-1], sig_print))

            if not np.any(location[0,t,:,0]): # this solves the problem of blank visualizations
                continue

            for m in range(M):
                x = location[m,t,:,0]  # (C,T,J,M)  #M, T, V, C 
                y = 1 - location[m,t,:,1]

                #print(M,t,x,y)

                col = []

                #thr = np.partition(result[int(frames[t]/4),:,m].flatten(), -show_n)[-show_n]
                bad_indices = []
                for v in range(V):
                    #r = r = result[int(frames[t]/4),v,m] >= thr if \
                    #    np.sum(result[int(frames[t]/4),:,m].flatten() >= thr) <= show_n else result[int(frames[t]/4),v,m]
                    

                    #if x[v] in bad_numbers or y[v] in bad_numbers:
                        

                    if sig == 0.0 and j_importances is not None:

                        #print("correct colors")
                        #r = 1 if result[t,v,m] > 0.3 else result[t,v,m]
                        r = result[m,t,v] #result[t,v,m]
                        r = r*3 if (r*3) < 1 else 1
                        g = 0.0 if r != 0 else 0.125
                        b = 1-r if r != 0 else 0.941#1-r
                        #r = 1.0 if r != 0 else 0.627

                    else:
                        g = 0.125
                        b = 0.941#1-r
                        r = 0.627

                    
                    k = connecting_joint[v] - 1

                    
                    bad_numbers = [0, 0.5]
                    
                    if (x[v] in bad_numbers and y[v] in bad_numbers) or y[v] == 0 or (x[k] in bad_numbers and y[k] in bad_numbers):
                        r = g= b = 1
                        #x[v] = y[v] = 0
                        #x = np.delete(x, v)
                        #y = np.delete(y, v)

                        bad_indices.append(v)
                        continue
                    
                    col.append([r, g, b])
                    plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=(0.1,0.1,0.1), linewidth=0.8, markersize=0)
                
                x = np.delete(x, bad_indices)
                y = np.delete(y, bad_indices)
                plt.scatter(x, y, marker='o', c=col, s=32)
            
            
            
            save_path = out_dir+ action_name + '/' + str(t) + '.png'
            plt.savefig(save_path, bbox_inches="tight")
            
            frame = cv2.imread(save_path) 

            if t == 0:
                video = cv2.VideoWriter(
                    filename=opts.checkpoint + "/" + action_name + "_sig" + str(sig) + pure + noisy + robust + masked + ".mp4", \
                        fourcc=fourcc, fps=fps, frameSize=(frame.shape[1],frame.shape[0])
                )

            video.write(frame)
        
        video.release()

    return new_printed


def val(model, test_loader, criterion, contrastive_loss, log_file_name):
    model.eval()
    batch_time = AverageMeter()
    total_losses = AverageMeter()
    ce_losses = AverageMeter()
    sc_losses = AverageMeter()
    cc_losses = AverageMeter()

    top1_base = AverageMeter()
    top5_base = AverageMeter()

    top1_masked = AverageMeter()
    top5_masked = AverageMeter()

    top1_noisy = AverageMeter()
    top5_noisy = AverageMeter()

    new_printed = 0
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

            ids = torch.arange(batch_size)
            #ids = torch.tensor([[0]])
            vis_ids = np.unique(ids)
            out_dir = opts.checkpoint + "/vis_mb/"
            #new_printed += show_skeleton(batch_gt, batch_input_pure, outputs[0],sig=0.0, ids=vis_ids, out_dir=out_dir, j_importances_org=j_importances[0])
            #output, j_importance, branch_inp = outputs[0], j_importances[0], branch_inps[0]

            ce_loss = criterion(outputs[0], batch_gt)

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

            if idx == 0:
                all_features = outputs[0].detach().cpu().numpy()  
                all_targets = batch_gt.detach().cpu().numpy()   
            else:
                all_features = np.vstack((all_features,outputs[0].detach().cpu().numpy()))
                all_targets = np.append(all_targets,batch_gt.detach().cpu().numpy()  )
                #print(all_targets.shape)
            # update metric
            total_losses.update(total_loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            sc_losses.update(sc_loss.item(), batch_size)
            cc_losses.update(cc_loss.item(), batch_size)

            acc1, acc5 = accuracy(outputs[0], batch_gt, topk=(1, 5))
            top1_base.update(acc1[0], batch_size)
            top5_base.update(acc5[0], batch_size)

            acc1, acc5 = accuracy(outputs[1], batch_gt, topk=(1, 5))
            top1_masked.update(acc1[0], batch_size)
            top5_masked.update(acc5[0], batch_size)

            acc1, acc5 = accuracy(outputs[2], batch_gt, topk=(1, 5))
            top1_noisy.update(acc1[0], batch_size)
            top5_noisy.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                log = str('Test: [{0}/{1}]\t'
                      'total_Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                      'ce_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'sc_Loss {sc_loss.val:.4f} ({sc_loss.avg:.4f})\t'
                      'cc_Loss {cc_loss.val:.4f} ({cc_loss.avg:.4f})\t'
                      'Acc@1_base {top1_base.val:.3f} ({top1_base.avg:.3f})\t'
                      'Acc@1_masked {top1_masked.val:.3f} ({top1_masked.avg:.3f})\t'
                      'Acc@1_noisy {top1_noisy.val:.3f} ({top1_noisy.avg:.3f})\t'.format(
                       idx, len(test_loader), total_loss = total_losses,
                       ce_loss=ce_losses, sc_loss=sc_losses, cc_loss=cc_losses, \
                        top1_base=top1_base, top1_masked=top1_masked, top1_noisy=top1_noisy))
                
                log_stuff(log_file_name, log)
                sys.stdout.flush()

            if new_printed == 10:
                quit()
            #break
    chk_filename = opts.evaluate if opts.evaluate else opts.resume
    filename = chk_filename.split(os.sep)[-1]
    filename_wo_ext = opts.checkpoint + os.sep + filename.split(".")[0] + "_" + args.dataset
    vis_latent_space(all_features, all_targets, filename_wo_ext)

    test_loss, test_top1_base, test_top5_base = total_losses.avg, top1_base.avg, top5_base.avg

    top1_masked, top5_masked = top1_masked.avg, top5_masked.avg

    top1_noisy, top5_noisy = top1_noisy.avg, top5_noisy.avg

    log = str('Loss {loss:.4f} \t'
            'Acc@1_base {top1_base:.3f} \t'
            'Acc@1_masked {top1_masked:.3f} \t'
            'Acc@1_noisy {top1_noisy:.3f} \t'
            'Acc@5_base {top5_base:.3f} \t'
            'Acc@5_masked {top5_masked:.3f} \t'
            'Acc@5_noisy {top5_noisy:.3f} \t'.format(loss=test_loss, top1_base=test_top1_base, \
                                                     top1_masked=top1_masked, top1_noisy=top1_noisy,
                                                     top5_base=test_top5_base, top5_masked=top5_masked, \
                                                        top5_noisy=top5_noisy))

    log_stuff(log_file_name, log)
    sys.stdout.flush()

    


    return test_top1_base, top1_masked, top1_noisy

def train_with_config(args, opts):
    print(args)

    log_file_name = opts.checkpoint + "/logs"+ ".txt"

    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    model_backbone = load_backbone(args)

    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    
    model = ActionNetppp(backbone=model_backbone,dim_rep=args.dim_rep, num_classes=args.action_classes,\
                        dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim,\
                            num_joints=args.num_joints, arch=args.head_version)

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

    testloader_params = {
          'batch_size': opts.batch_size,
          'shuffle': False,
          'num_workers': opts.num_workers,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    data_path = 'datasets/ntu60/Frames/%s.pkl' % args.dataset

    
    """ ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, \
                                random_move=False, scale_range=args.scale_range_test, chunk = opts.chunk)
    test_loader = DataLoader(ntu60_xsub_val, **testloader_params, drop_last=True) """


    chk_filename = opts.evaluate if opts.evaluate else opts.resume
    print('Loading checkpoint', chk_filename)

    model.model, checkpoint = load_checkpoint(model.model, chk_filename, clean=True) # model.model

    
    model = nn.DataParallel(model)
    model = model.cuda()

    sigmas = [0.0]
    """ sigmas = [0.00, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, \
                  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1] """

    import csv

    header = ['Sigma', 'Top1 Base', 'Top1 Masked', 'Top1 Noisy']

    with open(opts.checkpoint + '/results.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(header)

        for sigma in sigmas:
            ntu60_xsub_val = NTURGBD(data_path=data_path, data_split=args.data_split+'_val', n_frames=args.clip_len, \
                            random_move=False, scale_range=args.scale_range_test, chunk = opts.chunk, sigma=sigma)

            test_loader = DataLoader(ntu60_xsub_val, **testloader_params, drop_last=True)

            print('INFO: Testing on {} batches, printing output every {} batches'.format(len(test_loader), opts.print_freq))

            log_file_name = opts.checkpoint + "/logs_sig" +str(sigma)+ ".txt"

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("I will save the results in %s ....." %(log_file_name))
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            log = str('Sigma is  {sigma:.4f} poses have been clipped \t'.format(sigma=sigma))
            
            log_stuff(log_file_name, log)

            test_top1_base, top1_masked, top1_noisy= val(model, test_loader, criterion, contrastive_loss, log_file_name)

            data = [sigma, round(test_top1_base.item(), 2), round(top1_masked.item(),2), round(top1_noisy.item(),2)]
            writer.writerow(data)

            del ntu60_xsub_val, test_loader

                


    

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)