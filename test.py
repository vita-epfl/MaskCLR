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

from lib.model.DSTformer import *


from mask_me import MaskCollator

from utils.distributed import (
    init_distributed,
    AllReduce
)

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

batch_size = 4

N, M,T,J,C = batch_size,2,200,17,3

x = torch.rand(N, M,T,J,C).cuda()
#x = x.reshape(N*M, T, J, C)   



""" base_model = DSTformer()


out, _ = base_model(x.reshape(N*M,T,J,C))

print(out.shape) """

masker = MaskCollator()


masks_enc, masks_pred = masker.__call__(batch_size)

print("collated_masks_enc: ", masks_enc[0].shape)

print("collated_masks_pred: ", masks_pred[0].shape)

masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

encoder = DSTformerEncoder(dim_feat=256, maxlen=x.shape[2]).to(device)
predictor = DSTformerPredictor(depth=1, maxlen=x.shape[2],dim_in=encoder.dim_feat, dim_feat=128).to(device)

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

for m in encoder.modules():
    init_weights(m)

for m in predictor.modules():
    init_weights(m)

target_encoder = copy.deepcopy(encoder)

def forward_target():
    with torch.no_grad():
        h = target_encoder(x)

        print("h:", h.shape) # h: torch.Size([8, 256, 1280])
        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim

        print ("h after layer_norm: " , h.shape) # h after layer_norm:  torch.Size([8, 256, 1280])
        B = len(h)

        print ("Batch Size B: " , B) # Batch Size B:  8
        # -- create targets (masked regions of h)
        h = apply_masks(h, masks_pred)
        print("h after mask:", h.shape) # h after mask: torch.Size([32, 42, 1280])

        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))

        print("h after repeat_interleave_batch:", h.shape) # h after repeat_interleave_batch: torch.Size([32, 42, 1280])
        return h

def forward_context():
    z = encoder(x, masks_enc)

    print("z encoder out:", z.shape) # z encoder out: torch.Size([8, 85, 1280])

    z = predictor(z, masks_enc, masks_pred)

    print("z predictor out:", z.shape) # z predictor out: torch.Size([32, 42, 1280]) (batch_size*4, 42, 1280)
    return z

def loss_fn(z, h):
    loss = F.smooth_l1_loss(z, h)
    loss = AllReduce.apply(loss)
    return loss

# Step 1. Forward
with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
    h = forward_target()
    z = forward_context()
    loss = loss_fn(z, h)

print(loss)