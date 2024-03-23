import torch
import numpy as np
import pickle
import cv2
import math
import PIL
from PIL import Image
import torchvision.transforms.functional as F
import torch.nn.functional as f

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(file_path):
    img = np.load(file_path)
    img = img.astype(np.float32)
    img = img/255.
    img = img[:, :, [2, 1, 0]]
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img


def load_reflectPad(filepath):
    img = Image.open(filepath).convert('RGB')
    factor = 128
    h,w = img.size
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    img = f.pad(img, (0, padw, 0, padh), 'reflect')
    img = np.float32(img)
    img = img / 255.
    return img


def load_resize(filepath):
    img = Image.open(filepath).convert('RGB')
    wd_new, ht_new = img.size
    if ht_new > wd_new and ht_new > 1024:
        wd_new = int(np.ceil(wd_new * 1024 / ht_new))
        ht_new = 1024
    elif ht_new <= wd_new and wd_new > 1024:
        ht_new = int(np.ceil(ht_new * 1024 / wd_new))
        wd_new = 1024
    wd_new = int(128 * np.ceil(wd_new / 128.0))
    ht_new = int(128 * np.ceil(ht_new / 128.0))
    target_edge = wd_new if wd_new >= ht_new else ht_new
    img = img.resize((target_edge, target_edge), PIL.Image.ANTIALIAS)

    img = np.float32(img)
    img = img / 255.
    return img


def loader4dehaze(filepath,ps=256):
    img = Image.open(filepath).convert('RGB')
    w, h = img.size
    if w < ps :
        padW = 1+(ps-w)//2
        img = F.pad(img, (padW,0,padW,0), 0, 'constant')
    if h < ps :
        padH = 1+(ps-h)//2
        img = F.pad(img, (0,padH,0,padH), 0, 'constant')
    img = np.float32(img)
    img = img / 255.
    return img


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def myPSNR(tar_img, prd_img, cal_type):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)

    if cal_type == 'y':
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = imdff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        imdff = imdff.mul(convert).sum(dim=1)

    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True, cal_type='N'):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2, cal_type)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts

def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6))
    return score

def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=True)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img