import os
import time
import shutil
import random
import datetime
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_

from tcl_pyskl_dset.dataset import TclPysklDataset
from tcl_pyskl_dset.tcl_models import TSN
from tcl_pyskl_dset.transforms import *
from opts import parser
from ops import dataset_config
from tcl_pyskl_dset.utils import AverageMeter, accuracy
from tcl_pyskl_dset.temporal_shift import make_temporal_pool
import argparse
from mmcv import Config
from for_dataloader.builder import build_dataset, build_dataloader
from tcl_pyskl_dset.models import build_model

import torch
import sys

import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm

import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD
from lib.model.model_action import *


from mmcv.runner import build_optimizer

from matplotlib import pyplot as plt
from numpy import interp

import cv2
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle
from loss import SupConLoss

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

best_acc1 = 0
best_acc5 = 0
"""
CUDA_VISIBLE_DEVICES=0 python inference.py \
    configs/mb/MB_ft_NTU60_xsub.yaml \
    ntu RGB \
    --evaluate /home/ossamaabdelfattah/MotionBERT/checkpoint/mb_rc_of_val_fixed_t_t003.bin \
    --batch-size 8 \
    --exp_name exp/vis_pifpaf

"""

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

def main():
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    global args, best_acc1, best_acc5
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    if args.using_model == "mb":
        cfg.train_pipeline = None
        cfg.train_pipeline2 = None
        cfg.val_pipeline = None
    
    else:
        cfg.random_move= None
        cfg.scale_range_train = None
        cfg.scale_range_test = None


    ##asset check ####
    if args.use_finetuning:
        assert args.finetune_start_epoch > args.sup_thresh

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,args.modality)
    full_arch_name = args.arch
    if args.temporal_pool:
        full_arch_name += '_tpool'

    args.store_name = '_'.join(
        [args.exp_name, args.dataset, full_arch_name, 'p%.2f' % args.percentage, 'seg%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.dense_sample:
        args.store_name += '_dense'
    if args.non_local > 0:
        args.store_name += '_nl'
    if args.suffix is not None:
        args.store_name += '_{}'.format(args.suffix)
    print('storing name: ' + args.store_name)

    check_rootfolders()

    args.labeled_train_list, args.unlabeled_train_list=get_training_filenames(args.train_list)
    
    if (args.using_model == "tcl"):
        model = TSN(num_class, args.num_segments, args.modality,
                    base_model=args.arch,
                    consensus_type=args.consensus_type,
                    dropout=args.dropout,
                    img_feature_dim=args.img_feature_dim,
                    partial_bn=not args.no_partialbn,
                    pretrain=args.pretrain,
                    second_segments = args.second_segments,
                    is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                    fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                    temporal_pool=args.temporal_pool,
                    non_local=args.non_local,
                    n_channels=args.n_channels)
        
        policies = model.get_optim_policies()
    
    elif args.using_model == "mb":
        model_backbone = load_backbone(cfg)

        if cfg.finetune:
            if args.resume or args.evaluate:
                print("Not loading independent backbone..")
                pass
            else:
                #chk_filename = os.path.join(args.pretrained, args.selection)
                chk_filename = args.pretrained
                print('Loading backbone', chk_filename)
                checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
                model_backbone = load_pretrained_weights(model_backbone, checkpoint)

        if cfg.partial_train:
            model_backbone = partial_train_layers(model_backbone, cfg.partial_train)

        """ model = ActionNet(backbone=model_backbone, 
            dim_rep=cfg.dim_rep, 
            num_classes=cfg.action_classes, 
            dropout_ratio=cfg.dropout_ratio, 
            version=cfg.model_version, 
            hidden_dim=cfg.hidden_dim) """
        
        model = ActionNetppp(backbone=model_backbone,dim_rep=cfg.dim_rep, num_classes=cfg.action_classes,\
                        dropout_ratio=cfg.dropout_ratio, version=cfg.model_version, hidden_dim=cfg.hidden_dim,\
                            num_joints=cfg.num_joints)
        
        

        model_params = 0
        for parameter in model.parameters():
            model_params = model_params + parameter.numel()

        print('INFO: Trainable parameter count:', model_params)
        

    else:
        #assert args.n_channels == cfg.model["backbone"]["in_channels"]
        cfg.model["backbone"]["in_channels"] = args.n_channels
        model = build_model(cfg.model)
        optimizer = build_optimizer(model, cfg.optimizer)

        if (args.resume):
            policies = model.get_optim_policies()
            optimizer = torch.optim.SGD(policies,
                    args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
        
        

    print("Initialized Model = ", args.using_model)
    #print("==============model desccription=============")
    #print(model)

    

    

    if  args.evaluate:
        chk_filename = args.evaluate
        print('Loading checkpoint', chk_filename)

        model, checkpoint = load_checkpoint (model, chk_filename, clean=True)

        #for i in range(3):
        #    model.model_branches[i], _ = load_checkpoint(model.model_branches[i], chk_filename, clean=True)

    model = torch.nn.DataParallel(model).cuda()
    if args.temporal_pool and not args.resume and args.using_model == "tcl":
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True


    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    print("Done .... 5")


    if args.using_model != "mb":
        train_pipeline1_sampler = next(item for item in cfg.train_pipeline if item["type"] == "UniformSampleFrames") 
        train_pipeline1_sampler["clip_len"] = args.num_segments

        train_pipeline2_sampler = next(item for item in cfg.train_pipeline2 if item["type"] == "UniformSampleFrames") 
        train_pipeline2_sampler["clip_len"] = args.second_segments

        val_sampler = next(item for item in cfg.val_pipeline if item["type"] == "UniformSampleFrames") 
        val_sampler["clip_len"] = args.num_segments

        cfg.data["videos_per_gpu"]= args.batch_size

        if (args.n_channels ==17):
            assert next(item for item in cfg.train_pipeline if item["type"] == "FormatShape") is not None
            assert next(item for item in cfg.train_pipeline2 if item["type"] == "FormatShape") is not None

        

    cfg.seed = np.random.randint(2**31)
    

    val_dataset = TclPysklDataset(args.root_path, args.val_list, unlabeled=False,
                   num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=prefix,
                   random_shift=False,
                   second_segments = args.second_segments,
                   dense_sample=args.dense_sample, 
                   pipeline=cfg.val_pipeline, 
                   pipeline2=cfg.train_pipeline2,
                   sel_key = args.sel_key, 
                   use_tcl_sampling = args.use_tcl_sampling,
                   combine_skl_imgs= args.combine_skl_imgs,
                   bw_skl=args.bw_skl, 
                   bw_th=args.bw_th, 
                   img_feature_dim = args.img_feature_dim,
                   using_model=args.using_model,
                   random_move=False, 
                   scale_range=cfg.scale_range_test)

    if args.using_model == "mb":
        testloader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': False
        }

        val_loader = DataLoader(val_dataset, **testloader_params, drop_last=True)


    else:
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 4),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            seed=cfg.seed)
        dataloader_setting = dict(dataloader_setting,
                                    **cfg.data.get('train_dataloader', {}))


        dataloader_setting = dict(dataloader_setting,
                                    **cfg.data.get('val_dataloader', {}))
        val_loader = build_dataloader(val_dataset, **dataloader_setting)                   


    if args.using_model != "mb":
        for group in policies:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    #args.print_freq = args.batch_size // 10
    
    print("Evaluating the model...")
    validate(val_loader, model)
        

def show_skeleton(batch_gt, locations, out, j_importances,sig, ids, out_dir):

    batch_gt = batch_gt.cpu().detach().numpy()

    #print("locations.shape: ", locations.shape)

    locations = locations[:,:,:,:,:2]
    #locations = locations.permute(0,4,2,3,1)
    locations = locations.cpu().detach().numpy()

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

        result = j_importances[vid_idx,:,:,:]  # N, M, T, J


        #mm, tt, jj, _ = result.shape
        #result = np.reshape(result, ())

        #print("3: ", result.shape)

        probably_value = out[vid_idx, pred_class-1] #/100
        show_n = 6


        result = np.maximum(result, 0)
        result = result/np.max(result)

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
                        

                    if sig == 0.0:
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
                    filename=args.exp_name + "/" + action_name + "_sig" + str(sig) + pure + noisy + robust + masked + ".mp4", \
                        fourcc=fourcc, fps=fps, frameSize=(frame.shape[1],frame.shape[0])
                )

            video.write(frame)
        
        video.release()

    return new_printed


def validate(val_loader, model):

    # switch to evaluate mode
    model.eval()

    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = SupConLoss(temperature=0.07)
    criterion = criterion.cuda()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()

            #input = input.float()
            # compute output
            input, input_noisy = input

            outputs, features, j_importances, masked_inps = model(input, input_noisy)
            #output, features = output[1], features[0], j_importance[1]
            #output_noisy, _, _, _ = model(input_noisy)
            output_noisy = outputs[2]

            my_features = torch.vstack((features[0],features[0]))
            bsz = target.shape[0]

            f1, f2 = torch.split(my_features, [bsz, bsz], dim=0)

            my_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = criterion(my_features, target)

            #print( "loss: ", loss)

            #print(loss)

            ids = torch.arange(bsz)
            #ids = torch.tensor([[0]])
            vis_ids = np.unique(ids)

            features = features[0]
            features = features.detach().cpu().numpy()  
            targets = target.detach().cpu().numpy()          

            output, j_importance, masked_inp = outputs[0], j_importances[0], masked_inps[0]
            out = F.softmax(output, dim=1).detach().cpu().numpy()
            j_importance = j_importance.detach().cpu().numpy()
            out_dir = args.exp_name + "/vis_mb/"
            show_skeleton(target, masked_inp, out, j_importance,0.0, vis_ids, out_dir)

            output, j_importance, masked_inp = outputs[2], j_importances[2], masked_inps[2]
            out = F.softmax(output, dim=1).detach().cpu().numpy()
            j_importance = j_importance.detach().cpu().numpy()
            out_dir = args.exp_name + "/vis_mb_n/"
            #show_skeleton(target, masked_inp, out, j_importance,0.0, vis_ids, out_dir)
            
            if i == 0:
                all_features = features
                all_targets = targets
            else:
                all_features = np.vstack((all_features,features))
                all_targets = np.append(all_targets,targets)
                #print(all_targets.shape)

            #print(masked_inps[0].shape)

            

            if (args.using_model == "pyskl"):
                input = input["imgs"]

            # measure accuracy and record loss yeah = inpuy + 4
            acc1, acc5 = accuracy(outputs[2].data, target, topk=(1, 5))
            acc1_noisy, acc5_noisy = accuracy(output_noisy.data, target, topk=(1, 5))

            batch_size = len(input)  
            top1.update(acc1[0], batch_size)

            top5.update(acc5[0], batch_size)


            if i % args.print_freq == 0:
                output = ('Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                          'Acc_noisy@1 {top1_noisy:.3f}'.format(
                              i, len(val_loader),
                              top1=top1, top5=top5, top1_noisy=acc1_noisy.item()))
                print(output)
            
            #break
            if i==1:
                break 
        
    
    #print(all_features.shape, all_targets.shape)
    
    #vis_latent_space(all_features, all_targets, 'vis_features')
    
    output = ('Testing Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    
    print(output)

def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder) and 'vis' not in folder:
            print('creating folder ' + folder)
            os.makedirs(folder)



def split_file(file, unlabeled, labeled, percentage, isShuffle=True, seed=123, strategy='classwise'):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    if strategy == 'classwise':
        if os.path.exists(unlabeled) and os.path.exists(labeled):
          print("path exists with this seed and strategy")
          return 
        random.seed(seed)
        #creating dictionary against each category
        def del_list(list_delete,indices_to_delete):
            for i in sorted(indices_to_delete, reverse=True):
                del(list_delete[i])

        main_dict= defaultdict(list)
        with open(file,'r') as mainfile:
            lines = mainfile.readlines()
            for line in lines:
                video_info = line.strip().split()
                main_dict[video_info[2]].append((video_info[0],video_info[1]))
        
        print("Done 3....", unlabeled)
        with open(unlabeled,'w') as ul,\
            open(labeled,'w') as l:
            for key,value in main_dict.items():
                length_videos = len(value)
                ul_no_videos = int((length_videos* percentage))
                indices = random.sample(range(length_videos),ul_no_videos)
                for index in indices:
                    line_to_written = value[index][0] + " " + value[index][1] + " " +key+"\n"
                    ul.write(line_to_written)
                del_list(value,indices)
                for label_index in range(len(value)):
                    line_to_written = value[label_index][0] + " " + value[label_index][1] + " " +key+"\n"
                    l.write(line_to_written)



    if strategy == 'overall':
        if os.path.exists(unlabeled) and os.path.exists(labeled):
          print("path exists with this seed and strategy")
          return 
        random.seed(seed)
        with open(file, 'r') as fin, \
            open(unlabeled, 'w') as foutBig, \
            open(labeled, 'w') as foutSmall:
        # if didn't count you could only approximate the percentage
            lines = fin.readlines()
            random.shuffle(lines)
            nLines = sum(1 for line in lines)
            nTrain = int(nLines*percentage)
            i = 0
            for line in lines:
                line = line.rstrip('\n') + "\n"
                if i < nTrain:
                     foutBig.write(line)
                     i += 1
                else:
                     foutSmall.write(line)

def get_training_filenames(train_file_path):
    labeled_file_path = os.path.join("Run_"+str(int(np.round((1-args.percentage)*100))),args.dataset+'_'+str(args.seed)+args.strategy+"_labeled_training.txt")
    unlabeled_file_path = os.path.join("Run_"+str(int(np.round((1-args.percentage)*100))),args.dataset+'_'+str(args.seed)+args.strategy+"_unlabeled_training.txt")
    split_file(train_file_path, unlabeled_file_path,
               labeled_file_path,args.percentage, isShuffle=True,seed=args.seed, strategy=args.strategy)
    return labeled_file_path, unlabeled_file_path



if __name__ == '__main__':
    main()
