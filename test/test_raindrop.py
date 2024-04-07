import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

# import scipy.io as sio
from dataset.dataset_derain_drop import *
import utils
import math

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='Image derain evaluation on spad')
parser.add_argument('--input_dir', default='./dataset/AST_B/deraining/spad/val/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/deraining/spad/AST_B/se',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./logs/deraining/spad/AST_B/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='AST_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


utils.mkdir(args.result_dir)

test_dataset = get_test_data(args.input_dir)
# test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

model_restoration= utils.get_arch(args)

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()
model_restoration.eval()
from utils.image_utils import splitimage, mergeimage

def test_transform(v, op):
    v2np = v.data.cpu().numpy()
    if op == 'v':
        tfnp = v2np[:, :, :, ::-1].copy()
    elif op == 'h':
        tfnp = v2np[:, :, ::-1, :].copy()
    elif op == 't':
        tfnp = v2np.transpose((0, 1, 3, 2)).copy()

    ret = torch.Tensor(tfnp).to(v.device)

    return ret

def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask

# # Process data
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
        # rgb_noisy, mask = expand2square(data_test[1].cuda(), factor=128)
        filenames = data_test[2]
        
        input_ = data_test[1].cuda()
        B, C, H, W = input_.shape
        corp_size_arg = 384
        overlap_size_arg = 80
        split_data, starts = splitimage(input_, crop_size=corp_size_arg, overlap_size=overlap_size_arg)
        for i, data in enumerate(split_data):
            split_data[i] = model_restoration(data).cpu()
        restored = mergeimage(split_data, starts, crop_size=corp_size_arg, resolution=(B, C, H, W))
        rgb_restored = torch.clamp(restored, 0, 1).permute(0, 2, 3, 1).numpy()

        psnr = psnr_loss(rgb_restored[0], rgb_gt)
        ssim = ssim_loss(rgb_restored[0], rgb_gt, channel_axis=2, data_range=1)


        psnr_val_rgb.append(psnr)
        ssim_val_rgb.append(ssim)
        print("PSNR:",psnr,", SSIM:", ssim, filenames[0], rgb_restored.shape)
        utils.save_img(os.path.join(args.result_dir,filenames[0]+'.PNG'), img_as_ubyte(rgb_restored[0]))
        with open(os.path.join(args.result_dir,'psnr_ssim.txt'),'a') as f:
            f.write(filenames[0]+'.PNG ---->'+"PSNR: %.4f, SSIM: %.4f] "% (psnr, ssim)+'\n')
psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
print("PSNR: %f, SSIM: %f " %(psnr_val_rgb,ssim_val_rgb))
with open(os.path.join(args.result_dir,'psnr_ssim.txt'),'a') as f:
    f.write("Arch:"+args.arch+", PSNR: %.4f, SSIM: %.4f] "% (psnr_val_rgb, ssim_val_rgb)+'\n')