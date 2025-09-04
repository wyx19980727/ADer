from util.data import get_img_loader
from data.utils import get_transforms

import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm

import cv2 as cv
import matplotlib.pyplot as plt
import torch
import numpy as np


def calculate_fundamental_matrix(img1_path, img2_path):
	"""
	imgs1, imgs2: torch.Tensor, shape [B, N, 3, H, W]
	返回: F_list, shape [B, N, 3, 3] (每对图片一个F矩阵)
	"""
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
	return F

def get_epipolar_patch_masks(query_points, F, H_feat, W_feat, patch_size_h, patch_size_w):
    """
    对于每个query_point，获取其在img2极线上的所有patch的mask
    返回: masks, shape [num_query, H_feat, W_feat]，bool类型
    """
    device = query_points.device if isinstance(query_points, torch.Tensor) else 'cpu'
    num_query = H_feat * W_feat

    # img2 patch中心坐标
    y2_centers = torch.arange(H_feat, dtype=torch.float32, device=device) * patch_size_h + patch_size_h / 2 - 0.5  # [H_feat]
    x2_centers = torch.arange(W_feat, dtype=torch.float32, device=device) * patch_size_w + patch_size_w / 2 - 0.5  # [W_feat]
    v2_grid, u2_grid = torch.meshgrid(y2_centers, x2_centers, indexing='ij')  # [H_feat, W_feat]
    u2_grid = u2_grid.reshape(-1)  # [H_feat*W_feat]
    v2_grid = v2_grid.reshape(-1)  # [H_feat*W_feat]

    # query_points: [1, 2, H_feat, W_feat] -> [num_query, 3]
    x1 = query_points[0, 0].reshape(-1)  # [num_query]
    y1 = query_points[0, 1].reshape(-1)  # [num_query]
    ones = torch.ones_like(x1)
    pts1 = torch.stack([x1, y1, ones], dim=1)  # [num_query, 3]

    # F: numpy or torch, shape [3, 3]
    F_torch = torch.tensor(F, dtype=torch.float32, device=device) if not isinstance(F, torch.Tensor) else F

    # l = F @ pt1.T, 结果 shape [3, num_query]，转置后 [num_query, 3]
    l = torch.matmul(F_torch, pts1.t()).t()  # [num_query, 3]
    a = l[:, 0].unsqueeze(1)  # [num_query, 1]
    b = l[:, 1].unsqueeze(1)  # [num_query, 1]
    c = l[:, 2].unsqueeze(1)  # [num_query, 1]

    # 计算所有query和所有img2 patch中心的距离，利用广播
    # dist = |a*x + b*y + c| / sqrt(a^2 + b^2)
    # a,b,c: [num_query, 1], u2_grid/v2_grid: [1, H_feat*W_feat]
    dist = torch.abs(a * u2_grid + b * v2_grid + c) / (a.pow(2) + b.pow(2)).sqrt()  # [num_query, H_feat*W_feat]
    dist = dist.reshape(num_query, H_feat, W_feat)  # [num_query, H_feat, W_feat]

    # 距离小于阈值的patch视为在极线上
    masks = dist < max(patch_size_h, patch_size_w) / 2  # [num_query, H_feat, W_feat]
    return masks

# 0.1 preapare the two images

img1_path = '/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C2_20231021130235.jpg'
img2_path = '/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C3_20231021130235.jpg'

loader = get_img_loader('pil')

img1 = loader(img1_path)
img2 = loader(img2_path)

train_transforms = [
	dict(type='ToTensor'),
	dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
]

transform = get_transforms('', train=True, cfg_transforms=train_transforms)

img1 = transform(img1)
img1 = img1.unsqueeze(0)  # Add batch dimension [1, 3, 256, 256]
img2 = transform(img2)
img2 = img2.unsqueeze(0)  # Add batch dimension [1, 3, 256, 256]

# 0.2 preapare the teacher features
model = timm.create_model('resnet18', pretrained=True, features_only=True, out_indices=(2,))

img1_teacher_feature = model(img1)[0] # [1, 128, 32, 32]
img2_teacher_feature = model(img2)[0]

# 0.3 calculate the fundamental matrix (NEED IMPROVEMENT TO TORCH GPU VERSION)
Fundamental_matrix = calculate_fundamental_matrix(img1_path, img2_path)

# 0.4 prepare all the query points of all the img1 patches
B, C, H_feat, W_feat = img1_teacher_feature.shape  # [1, 128, 32, 32]
H_img, W_img = img1.shape[2], img1.shape[3]        # 256, 256

# 计算每个patch的中心点在输入图像上的坐标
# 假设patch均匀划分，patch_size = H_img // H_feat = 8
patch_size_h = H_img // H_feat
patch_size_w = W_img // W_feat

# 计算中心点坐标（torch实现）
y_centers = torch.arange(H_feat, dtype=torch.float32) * patch_size_h + patch_size_h / 2 - 0.5  # [32]
x_centers = torch.arange(W_feat, dtype=torch.float32) * patch_size_w + patch_size_w / 2 - 0.5  # [32]

# 生成网格
v_coord, u_coord = torch.meshgrid(y_centers, x_centers, indexing='ij')  # [32, 32]，分别是y和x

# 组合成 [1, 2, 32, 32]
query_points = torch.stack([u_coord, v_coord], dim=0).unsqueeze(0)  # [1, 2, 32, 32]

# 1. 获取每个query_point在img2极线上的所有patches的mask(维度为[32*32, 32, 32])
masks = get_epipolar_patch_masks(query_points, Fundamental_matrix, H_feat, W_feat, patch_size_h, patch_size_w)

# 2. 对于每个img1 query_point对应的query embedding和mask，获取img2中对应的patch embeddings，并使用torch scaled_dot_product_attention function实现cross view attention
# img1_teacher_feature: [1, 128, 32, 32]
# img2_teacher_feature: [1, 128, 32, 32]
# masks: [1024, 32, 32]
num_query = H_feat * W_feat

# [1024, 128]
query_embeddings = img1_teacher_feature[0].reshape(C, -1).transpose(0, 1)  # [num_query, C]
img2_patches = img2_teacher_feature[0].reshape(C, -1).transpose(0, 1)      # [num_patch, C] (1024, 128)

# [num_query, 1024]，每一行是该query的mask
masks_flat = masks.reshape(num_query, -1)  # [num_query, 1024]

# 构造Q,K,V
Q = query_embeddings.unsqueeze(1)  # [num_query, 1, C]
K = img2_patches.unsqueeze(0).expand(num_query, -1, -1)  # [num_query, 1024, C]
V = K  # [num_query, 1024, C]

# 对mask为False的位置赋极小值，防止被softmax
attn_mask = ~masks_flat  # [num_query, 1024], True为需要mask掉
# scaled_dot_product_attention 支持 attn_mask: [num_query, 1, 1024]
attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask.unsqueeze(1))  # [num_query, 1, C]
attn_out = attn_out.squeeze(1)  # [num_query, C]

# 若某query没有极线patch（全mask），其attn_out为nan，可用原特征替换
nan_mask = torch.isnan(attn_out).any(dim=1)
attn_out[nan_mask] = query_embeddings[nan_mask]

# 恢复回[1, 128, 32, 32]
enhanced_query_features = attn_out.transpose(0, 1).reshape(1, C, H_feat, W_feat)

import ipdb; ipdb.set_trace()
