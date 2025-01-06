import os

import cv2
import random
import numpy as np
import argparse


interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    return np.random.choice(interp_list)
    # return cv2_interp


def alpha2mask(pha):
    erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]
    # alpha_ori = sample['alpha']
    h, w = pha.shape

    max_kernel_size = 30
    alpha = cv2.resize(pha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

    low = 0.01
    high = 1.0
    thres = random.random() * (high - low) + low
    seg_mask = (alpha >= thres).astype(np.int).astype(np.uint8)
    random_num = random.randint(0, 3)
    if random_num == 0:
        seg_mask = cv2.erode(seg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])
    elif random_num == 1:
        seg_mask = cv2.dilate(seg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])
    elif random_num == 2:
        seg_mask = cv2.erode(seg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])
        seg_mask = cv2.dilate(seg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])
    elif random_num == 3:
        seg_mask = cv2.dilate(seg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])
        seg_mask = cv2.erode(seg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])

    seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST) * 255.

    return seg_mask


def alpha2trimap(pha):
    erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]
    alpha_ori = pha
    h, w = alpha_ori.shape
    alpha = cv2.resize(alpha_ori, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

    fg_width = np.random.randint(1, 30)
    bg_width = np.random.randint(1, 30)
    # fg_width = np.random.randint(5, 15)
    # bg_width = np.random.randint(5, 15)
    # fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
    fg_mask = (alpha + 1e-5).astype(int).astype(np.uint8)
    # bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
    bg_mask = (1 - alpha + 1e-5).astype(int).astype(np.uint8)
    fg_mask = cv2.erode(fg_mask, erosion_kernels[fg_width])
    bg_mask = cv2.erode(bg_mask, erosion_kernels[bg_width])

    trimap = np.ones_like(alpha) * 128
    trimap[fg_mask == 1] = 255
    trimap[bg_mask == 1] = 0

    trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)

    return trimap


def parse_args():
    parser = argparse.ArgumentParser(description='Convert alpha matte to trimap')
    parser.add_argument('--alpha_path', type=str, required=True,
                      help='Path to the alpha matte directory')
    parser.add_argument('--trimap_path', type=str, required=True,
                      help='Path to save the generated trimaps')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create trimap directory if it doesn't exist
    os.makedirs(args.trimap_path, exist_ok=True)
    
    for image_name in os.listdir(args.alpha_path):
        print('Processing ' + image_name)
        image_path = os.path.join(args.alpha_path, image_name)
        img = cv2.imread(image_path, 0) / 255.
        trimap = alpha2trimap(img)
        trimap_path = os.path.join(args.trimap_path, image_name)
        cv2.imwrite(trimap_path, trimap)

