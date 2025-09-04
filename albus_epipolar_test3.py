import cv2 as cv
import numpy as np

from util.data import get_img_loader
from data.utils import get_transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# 1. Fundamental matrix 1: cv2 load

img1_path = '/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C2_20231021130235.jpg'
img2_path = '/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C3_20231021130235.jpg'

img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN parameters
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
pts1 = []
pts2 = []
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)
print("Fundamental Matrix 1 cv2 load F:\n", F)

# 2. Fundamental matrix 2: imagenet normalize
loader = get_img_loader('pil')

img1 = loader(img1_path)
img2 = loader(img2_path)

train_transforms = [
	dict(type='ToTensor'),
	dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
]

transform = get_transforms('', train=True, cfg_transforms=train_transforms)

img1 = transform(img1)
img2 = transform(img2)

def denormalize_img(img_tensor, mean, std):
    # img_tensor: [C, H, W], float, 0~1或-均值/标准差
    img = img_tensor.clone().cpu()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    img = img.permute(1, 2, 0).numpy()  # [H, W, C]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img

# 还原为可用于SIFT的灰度图
img1_denorm = denormalize_img(img1, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
img2_denorm = denormalize_img(img2, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
img1_gray = cv.cvtColor(img1_denorm, cv.COLOR_RGB2GRAY)
img2_gray = cv.cvtColor(img2_denorm, cv.COLOR_RGB2GRAY)

import ipdb; ipdb.set_trace()

# SIFT和FLANN流程同前
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
good = []
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F2, mask2 = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)
print("Fundamental Matrix 2 (imagenet normalized img):\n", F2)