"""
partial codes are borrowed from CycleISP repo:
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""
import torch
import glob
import os
import rawpy
import pickle
import cv2
import argparse
import numpy as np
from noise_sampling import random_noise_levels_dnd, random_noise_levels_sidd, add_noise
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/data2/PASCALRAW/original/raw/')
parser.add_argument('--save_dir', default='/10T-disk/low-light-noise-jpg-PASCALRAW')
parser.add_argument('--EV', default=0, type=int)
args = parser.parse_args()


exposure = args.EV
bit_depth = 12
JPG_DIR = os.path.join(args.save_dir, f'ev_{exposure}', 'jpg')
RAW_DIR = os.path.join(args.save_dir, f'ev_{exposure}', 'raw')
PARAM_DIR = os.path.join(args.save_dir, f'ev_{exposure}', 'parameter')
for dir_name in [JPG_DIR, RAW_DIR, PARAM_DIR]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


with torch.no_grad():

    for filename in tqdm(sorted(glob.glob(os.path.join(args.input_dir, '*.nef')))):
        raw = rawpy.imread(filename)
        black_level = raw.black_level_per_channel[0] # assume they're all the same
        # First do low light
        im = raw.raw_image.astype(np.float32)
        im = np.maximum(im, black_level) - black_level # changed order of computation
        im = (im * (2**exposure))
        im = im + black_level
        im = np.minimum(im, 2**bit_depth - 1)
        im = im / (2**bit_depth - 1)

        # Second add noise(use code from CycleISP) input raw need to be [0,1]
        low_light_raw = torch.from_numpy(im).cuda()
        shot_noise, read_noise = random_noise_levels_dnd()
        shot_noise, read_noise = shot_noise.cuda(), read_noise.cuda()
        raw_noisy = add_noise(low_light_raw, shot_noise, read_noise, use_cuda=True)
        raw_noisy = torch.clamp(raw_noisy,0,1)  ### CLIP NOISE
        variance = shot_noise * raw_noisy + read_noise
        raw.raw_image[:] = (raw_noisy.cpu().detach().numpy() * (2 ** bit_depth -1)).astype(np.uint16)
        # print(raw.raw_image.min(), raw.raw_image.max())
        noise_jpg = raw.postprocess(use_camera_wb=True, no_auto_bright=True)

        # save file {JPG, RAW, PARAM}
        basename = os.path.basename(filename)
        path_jpg = os.path.join(JPG_DIR, basename.replace('nef', 'jpg'))
        cv2.imwrite(path_jpg, noise_jpg[:, :, ::-1])

        path_raw = os.path.join(RAW_DIR, basename.replace('nef', 'pkl'))
        with open(path_raw, 'wb') as f:
            pickle.dump(raw.raw_image, f)

        '''
        not save parameter because of the size (3xx GiB)
        path_param = os.path.join(PARAM_DIR, basename.replace('nef', 'pkl'))
        d = dict()
        d['variance'] = variance.cpu().detach().numpy()
        d['shot_noise'] = shot_noise.cpu().detach().numpy()
        d['read_noise'] = read_noise.cpu().detach().numpy()
        with open(path_param, 'wb') as f:
            pickle.dump(d, f)
        '''
