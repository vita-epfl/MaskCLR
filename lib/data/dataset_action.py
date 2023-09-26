import torch
import numpy as np
import os
import random
import copy
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl

from tqdm import tqdm

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()

def random_frame_mask(motion, max_len):
    
    x = copy.deepcopy(motion)

    masked_frames_num = np.random.randint(0, max_len//2)

    masked_frames_inds = random.sample(range(0, max_len), masked_frames_num)

    x = np.delete(x, masked_frames_inds, axis=1)

    return x


def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y
    
def random_move(motion,
                angle_range=[-10., 10.],
                scale_range=[0.9, 1.1],
                transform_range=[-0.1, 0.1],
                move_time_candidate=[1]):
    
    data_numpy = copy.deepcopy(motion)

    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # C,T,V,M -> M,T,V,C
    return data_numpy    

def human_tracking(x):
    M, T = x.shape[:2]
    if M==1:
        return x
    else:
        diff0 = np.sum(np.linalg.norm(x[0,1:] - x[0,:-1], axis=-1), axis=-1)        # (T-1, V, C) -> (T-1)
        diff1 = np.sum(np.linalg.norm(x[0,1:] - x[1,:-1], axis=-1), axis=-1)
        x_new = np.zeros(x.shape)
        sel = np.cumsum(diff0 > diff1) % 2
        sel = sel[:,None,None]
        x_new[0][0] = x[0][0]
        x_new[1][0] = x[1][0]
        x_new[0,1:] = x[1,1:] * sel + x[0,1:] * (1-sel)
        x_new[1,1:] = x[0,1:] * sel + x[1,1:] * (1-sel)
        return x_new

def preprocess_data(sample_org, num_segments, is_train, noise = False):
    sample = copy.deepcopy(sample_org)

    resample_id = resample(ori_len=sample['total_frames'], target_len=num_segments, randomness=is_train)

    #print("resample_id.shape: ", resample_id.shape)

    # resample_id (48,)
    # Before make_cam (1, 103, 17, 2)
    motion_cam = make_cam(x=sample['keypoint'], img_shape=sample['img_shape'])
    # After make_cam (1, 103, 17, 2)

    #print("motion_cam.shape", motion_cam.shape)

    if noise:
        sigma = 0.003
        _, T, V, M = motion_cam.shape
        noise = np.random.normal(0, sigma , (T, V, M))
        motion_cam[:,:,:,:] += noise
        motion_cam = np.clip(motion_cam, -1, 1)
    

    

    motion_cam = human_tracking(motion_cam)
    # After human_tracking (1, 103, 17, 2)

    motion_cam = coco2h36m(motion_cam)
    # After coco2h36m (1, 103, 17, 2)

    motion_conf = sample['keypoint_score'][..., None]
    motion = np.concatenate((motion_cam[:,resample_id], motion_conf[:,resample_id]), axis=-1)
    # After sample['keypoint_score'] (1, 48, 17, 3)

    if motion.shape[0]==1:                                  # Single person, make a fake zero person
        fake = np.zeros(motion.shape)
        motion = np.concatenate((motion, fake), axis=0)
    
    #print(motion.shape)

    return motion



class ActionDataset(Dataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1],\
                  check_split=True, sigma=0.0, of = False, chunk = None):   # data_split: train/test etc.
        np.random.seed(0)
        dataset = read_pkl(data_path)
        #data_path_noisy = "datasets/ntu60/Frames/ntu60_hrnet_noisy0.003.pkl"

        #print("reading... " , data_path_noisy)

        #dataset_noisy = read_pkl(data_path_noisy)

        filename = data_path.split(os.sep)[-1]
        dataset_name = filename.split(".")[0]

        #chucnk = 40 if 'train' in data_split else 10


        if check_split:
            assert data_split in dataset['split'].keys()
            self.split = dataset['split'][data_split]#[:chucnk]

            #del self.split[1::2]  # delete odd indexed items


            #print(type(chunk), chunk)
            self.split = self.split[:chunk] if chunk is not None else self.split

        annotations = dataset['annotations']
        #annotations_noisy = dataset_noisy['annotations']#[:chunk]

        self.random_move = random_move
        self.is_train = "train" in data_split or (check_split==False)
        if "oneshot" in data_split:
            self.is_train = False
        self.scale_range = scale_range
        motions = [[],[]]
        labels = []

        if check_split:
            print("extracting %s items of %s out of the %s items of the dataset, adding noise sig = %s" 
                % (str(len(self.split )), str(data_split), str(len(annotations)), str(sigma)))

        #check_split = False
        for sample in annotations:
            if (check_split and (not sample['frame_dir'] in self.split) and not of):
                continue
            
            motion = preprocess_data(sample, n_frames, self.is_train, noise = False)

            motion_noisy = preprocess_data(sample, n_frames, self.is_train, noise = True)

            motions[0].append(motion.astype(np.float32)) 
            motions[1].append(motion_noisy.astype(np.float32)) 
            labels.append(sample['label'])
        self.motions = np.array(motions[0])
        self.motions_noisy = np.array(motions[1])
        self.labels = np.array(labels)
        
        print("extracted %s items for %s!" % (len(self.labels), str(data_split)))

        print("preprocessing done!")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)

    def __getitem__(self, index):
        raise NotImplementedError 

class NTURGBD(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], sigma=0.0, of=False, chunk = None, check_split=True):
        super(NTURGBD, self).__init__(data_path, data_split, n_frames, random_move, scale_range, sigma=sigma, of=of, chunk = chunk)
        

        self.n_frames = n_frames
        #print("Sigma here is: ", sigma)
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, motion_noisy, label = self.motions[idx],self.motions_noisy[idx], self.labels[idx] # (M,T,J,C)

        if self.random_move:
            motion = random_move(motion)
            motion_noisy = random_move(motion_noisy)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
            result_noisy = crop_scale(motion_noisy, scale_range=self.scale_range)
        else:
            result = motion
            result_noisy = motion_noisy

        #result = random_frame_mask(result, self.n_frames)

        #print("result.shape: ", result.shape)

        return [[result.astype(np.float32), result_noisy.astype(np.float32)], label]
    
class NTURGBD1Shot(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], check_split=False):
        super(NTURGBD1Shot, self).__init__(data_path, data_split, n_frames, random_move, scale_range, check_split)
        oneshot_classes = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114]
        new_classes = set(range(120)) - set(oneshot_classes)
        old2new = {}
        for i, cid in enumerate(new_classes):
            old2new[cid] = i
        filtered = [not (x in oneshot_classes) for x in self.labels]
        self.motions = self.motions[filtered]
        filtered_labels = self.labels[filtered]
        self.labels = [old2new[x] for x in filtered_labels]
        
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        
        return result.astype(np.float32), label