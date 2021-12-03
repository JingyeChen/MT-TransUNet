import sys
sys.path.append('/home/db/Joint-seg-cls-jhu/model')
sys.path.append('/home/db/Joint-seg-cls-jhu/model/network')

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling_combine import VisionTransformer as ViT_seg
# from networks.vit_seg_modeling_combine_decoder_add_image import VisionTransformer as ViT_seg
# from networks.vit_seg_modeling_transkip import VisionTransformer as ViT_seg
# from networks.vit_seg_modeling_transkip_all import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

class args:
    root_path = '../data/Synapse/train_npz'
    dataset = 'Synapse'
    list_dir = './lists/lists_Synapse',
    num_classes = 2
    max_iterations = 30000
    max_epochs = 150
    batch_size = 24
    n_gpu = 1
    deterministic = 1
    base_lr = 0.01
    img_size = 224
    seed = 1234
    n_skip = 3
    vit_name = 'R50-ViT-B_16'
    # vit_name = 'R50-ViT-L_16'
    vit_patches_size = 16

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
dataset_name = args.dataset

args.is_pretrain = True
args.exp = 'TU_' + dataset_name + str(args.img_size)

config_vit = CONFIGS_ViT_seg[args.vit_name]
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
net.load_from(weights=np.load(config_vit.pretrained_path))
