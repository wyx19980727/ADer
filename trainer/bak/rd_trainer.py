import os
import copy
import glob
import shutil
import datetime
import time

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
    from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp
import cv2 as cv

real_iad_classes = [
    'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
    'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery',
    'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
    'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
    'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
    'usb_adaptor', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
@TRAINER.register_module
class RDTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(RDTrainer, self).__init__(cfg)
        
    def get_all_epipolar_pixels(self, F, img1_shape, img2_shape):
        """
        获取img1上所有像素点在img2上对应的极线上所有像素点
        F: 基础矩阵 [3, 3]
        img1_shape: (height, width)
        img2_shape: (height, width)
        返回: 
            dict: {
                (y1, x1): [(y2, x2), ...]  # img1像素点 => img2上极线像素列表
            }
        """
        H1, W1 = img1_shape
        H2, W2 = img2_shape
        
        # 生成img1所有像素坐标网格 [H1, W1, 2]
        y1_coords, x1_coords = np.mgrid[0:H1, 0:W1]
        p1_coords = np.stack((x1_coords, y1_coords, np.ones_like(x1_coords)), axis=-1)  # [H1, W1, 3]
        
        # 计算所有点在img2上的极线 [H1, W1, 3]
        l2 = np.einsum('ij,klj->kli', F, p1_coords)  # 向量化计算
        
        # 准备img2的像素网格 [H2, W2]
        y2_grid, x2_grid = np.mgrid[0:H2, 0:W2]
        
        # 结果字典初始化
        epipolar_dict = {}
        
        # 遍历img1的每个像素
        for y1 in range(H1):
            for x1 in range(W1):
                a, b, c = l2[y1, x1]
                pixels_on_line = []
                
                # 处理垂直线 (b ≈ 0)
                if abs(b) < 1e-5:
                    if abs(a) > 1e-5:
                        x = -c / a
                        if 0 <= x < W2:
                            for y in range(H2):
                                pixels_on_line.append((int(y), int(x)))
                else:
                    # 计算该极线与img2所有行的交点
                    for y2 in range(H2):
                        x = -(b * y2 + c) / a if abs(a) > 1e-5 else 0
                        if 0 <= x < W2:
                            pixels_on_line.append((int(y2), int(x)))
                
                epipolar_dict[(y1, x1)] = pixels_on_line
        
        return epipolar_dict
        
    def set_input(self, inputs):
        # self.imgs = inputs['img'].cuda()
        # self.imgs_mask = inputs['img_mask'].cuda()
        # self.cls_name = inputs['cls_name']
        # self.anomaly = inputs['anomaly']
        # self.bs = self.imgs.shape[0]
        # import ipdb; ipdb.set_trace()
        self.imgs = inputs['img'].cuda()
        index_combinations = torch.combinations(torch.arange(5), r=2)
        self.imgs1 = self.imgs[:, index_combinations[:, 0], :, :, :]
        self.imgs2 = self.imgs[:, index_combinations[:, 1], :, :, :]
        F = self.calculate_fundamental_matrix(self.imgs1, self.imgs2)
        #import ipdb; ipdb.set_trace()
  
        img1_tensor = self.imgs1[0, 1] # 取 batch=0, pair=1 的第一张图
        img2_tensor = self.imgs2[0, 1] # 取 batch=0, pair=1 的第二张图
        F_single = F[0, 1] # 取 batch=0, pair=1 的F矩阵
        
        # 获取图像尺寸
        _, H1, W1 = img1_tensor.shape
        _, H2, W2 = img2_tensor.shape
        
        # 计算所有像素的极线对应关系
        epipolar_map = self.get_all_epipolar_pixels(
            F_single, 
            img1_shape=(H1, W1),
            img2_shape=(H2, W2)
        )
        
        center_y, center_x = H1 // 2, W1 // 2
        center_epipolar = epipolar_map.get((center_y, center_x), [])
        print(f"({center_x},{center_y}) contains {len(center_epipolar)} pixels on the epipolar line in img2")
        
        import ipdb; ipdb.set_trace()
        
        
  
        # # 将Tensor转换为OpenCV格式的Numpy数组
        # img1_np = img1_tensor.cpu().numpy().transpose(1, 2, 0)
        # img2_np = img2_tensor.cpu().numpy().transpose(1, 2, 0)
        # img1_np = (img1_np * 255).astype(np.uint8) if img1_np.max() <= 1.0 else img1_np.astype(np.uint8)
        # img2_np = (img2_np * 255).astype(np.uint8) if img2_np.max() <= 1.0 else img2_np.astype(np.uint8)
  
        # # 如果是BGR顺序，需要转换
        # img1_np = cv.cvtColor(img1_np, cv.COLOR_RGB2BGR)
        # img2_np = cv.cvtColor(img2_np, cv.COLOR_RGB2BGR)
  
        # visualization, epilines_in_img2 = self.find_and_draw_epilines(img1_np, img2_np, F_single)
        # print("epipolar line parameters (a,b,c):")
        # print(epilines_in_img2)
        # cv.imwrite('epipolar_visualization.jpg', visualization)
        
        # w, h = img2_np.shape[1], img2_np.shape[0]
        # line_pixels = self.get_pixels_on_line(epilines_in_img2[0], w, h)
        # print("coordinates of pixels on the epipolar line:")
        # for pixel in line_pixels:
        #     print(pixel)
        
        import ipdb; ipdb.set_trace()


        
  
        self.imgs_mask = inputs['img_mask'].cuda()
        if len(self.imgs.shape)==5:
            self.imgs = self.imgs.flatten(0, 1)
            self.imgs_mask = self.imgs_mask.flatten(0, 1)
        self.cls_name = inputs['cls_name']
        self.cls_name = np.array(self.cls_name)
        self.cls_name = np.transpose(self.cls_name, (1, 0)) 
        #class_to_index = {cls: idx for idx, cls in enumerate(real_iad_classes)}
        #import ipdb; ipdb.set_trace()
        #self.contrast_labels = np.tile(np.arange(self.cls_name.shape[1]), (self.cls_name.shape[0], 1)).flatten()
        #self.contrast_labels = np.array([class_to_index[item] for item in self.cls_name.flatten()])
        
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        #import ipdb; ipdb.set_trace()
        self.img_path = np.array(self.img_path)
        self.img_path = np.transpose(self.img_path, (1, 0))
        self.img_path = self.img_path.flatten()
  
        self.sample_anomaly = inputs['sample_anomaly']
        self.bs = self.imgs.shape[0]
  
    # def calculate_fundamental_matrix(self, imgs1, imgs2):
    # 	sift = cv.xfeatures2d.SIFT_create()
    # 	# find the keypoints and descriptors with SIFT
    # 	kp1, des1 = sift.detectAndCompute(imgs1, None)
    # 	kp2, des2 = sift.detectAndCompute(imgs2, None)
    # 	# FLANN parameters
    # 	FLANN_INDEX_KDTREE = 1
    # 	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # 	search_params = dict(checks=50)
    # 	flann = cv.FlannBasedMatcher(index_params, search_params)
    # 	matches = flann.knnMatch(des1, des2, k=2)
    # 	good = []
    # 	pts1 = []
    # 	pts2 = []
      # 	# ratio test as per Lowe's paper
    # 	for i, (m, n) in enumerate(matches):
    # 		if m.distance < 0.8 * n.distance:
    # 			good.append(m)
    # 			pts2.append(kp2[m.trainIdx].pt)
    # 			pts1.append(kp1[m.queryIdx].pt)
    # 	pts1 = np.int32(pts1)
    # 	pts2 = np.int32(pts2)
    # 	F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_8POINT)
    # 	return F
 
    

    def calculate_fundamental_matrix(self, imgs1, imgs2):
        """
        imgs1, imgs2: torch.Tensor, shape [B, N, 3, H, W]
        返回: F_list, shape [B, N, 3, 3] (每对图片一个F矩阵)
        """
        B, N, C, H, W = imgs1.shape
        F_list = []
        sift = cv.xfeatures2d.SIFT_create()
        for b in range(B):
            F_list_b = []
            for n in range(N):
                img1 = imgs1[b, n].cpu().numpy().transpose(1, 2, 0)  # [3, H, W] -> [H, W, 3]
                img2 = imgs2[b, n].cpu().numpy().transpose(1, 2, 0)
                img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
                img2 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
                gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
                gray2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
                kp1, des1 = sift.detectAndCompute(gray1, None)
                kp2, des2 = sift.detectAndCompute(gray2, None)
                F = None
                if des1 is not None and des2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2)
                    good = []
                    pts1 = []
                    pts2 = []
                    for m_n in matches:
                        if len(m_n) == 2:
                            m, n = m_n
                            if m.distance < 0.8 * n.distance:
                                good.append(m)
                                pts2.append(kp2[m.trainIdx].pt)
                                pts1.append(kp1[m.queryIdx].pt)
                    if len(pts1) >= 8 and len(pts2) >= 8:
                        pts1 = np.int32(pts1)
                        pts2 = np.int32(pts2)
                        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)
                F_list_b.append(F if F is not None else np.zeros((3, 3)))
            F_list.append(np.stack(F_list_b))
        return np.stack(F_list)  # shape [B, N, 3, 3]

    
    def draw_epilines(self, img1, img2, lines, pts1, pts2):
        """
        在图像上绘制极线和对应的特征点
        img1: 绘制特征点pts1的图像
        img2: 绘制极线和特征点pts2的图像
        lines: 对应pts1在img2上的极线列表
        pts1: img1上的特征点
        pts2: img2上的特征点
        """
        #import ipdb; ipdb.set_trace()
        r, c = img1.shape[:2] # 获取图像的行和列
        # 确保图像是彩色图，以便绘制彩色的线
        if len(img1.shape) == 2:
            img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        if len(img2.shape) == 2:
            img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

        # 遍历每一条极线和对应的点
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # r = [a, b, c], 极线方程 a*x + b*y + c = 0
            a, b, c = r[0], r[1], r[2]
            
            # 计算极线在图像边界上的两个端点，以便绘制
            x0, y0 = map(int, [0, -c / b])
            x1, y1 = map(int, [c, -(c + a * c) / b])
            
            # 绘制极线
            img2 = cv.line(img2, (x0, y0), (x1, y1), color, 1)
            # 绘制特征点
            img1 = cv.circle(img1, tuple(map(int, pt1)), 5, color, -1)
            img2 = cv.circle(img2, tuple(map(int, pt2)), 5, color, -1)
            
        return img1, img2
 
    def find_and_draw_epilines(self, img1, img2, F):
        """
        主函数：找到关键点、计算F矩阵，并计算和绘制极线

        img1: 第一张图片 (numpy array, uint8)
        img2: 第二张图片 (numpy array, uint8)
        F: 已经计算好的基本矩阵
        """
        # 将图像转为灰度图
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        # --- 以下部分与您的 calculate_fundamental_matrix 类似，用于获取匹配点 ---
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        pts1 = []
        pts2 = []
        good = []
        # Lowe's ratio test
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                good.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        # --- 匹配点获取结束 ---
        
        # === 核心步骤: 计算极线 ===
        # 1. 计算 img1 中的点在 img2 上的极线
        # whichImage=1 表示 pts1 是第一张图的点，我们要计算它们在第二张图上的线
        lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3) # a, b, c
        
        # 2. 计算 img2 中的点在 img1 上的极线
        # whichImage=2 表示 pts2 是第二张图的点，我们要计算它们在第一张图上的线
        lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3) # a, b, c
        
        # 绘制极线
        img_with_lines2, _ = self.draw_epilines(img1, img2, lines2, pts1, pts2)
        img_with_lines1, _ = self.draw_epilines(img2, img1, lines1, pts2, pts1)
        #import ipdb; ipdb.set_trace()
        
        # 将两张图拼接起来显示
        h1, w1 = img_with_lines1.shape[:2]
        h2, w2 = img_with_lines2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        vis[:h1, :w1] = img_with_lines1
        vis[:h2, w1:w1 + w2] = img_with_lines2

        # cv2.imshow('Epilines', vis)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return vis, lines2
 
    def get_pixels_on_line(self, line_abc, width, height):
        """
        根据极线方程 [a, b, c] 和图像尺寸，获取线上所有像素的坐标
        """
        a, b, c = line_abc
        pixels = []

        # 遍历图像的每一列
        for x_prime in range(width):
            # 根据 a*x' + b*y' + c = 0 计算 y'
            # 我们需要处理 b 约等于 0 的情况 (垂直线)
            if abs(b) > 1e-6:
                y_prime = int((-a * x_prime - c) / b)
                # 确保计算出的点在图像范围内
                if 0 <= y_prime < height:
                    pixels.append((x_prime, y_prime))
        
        # 如果是垂直线，b=0, a*x'+c=0 => x' = -c/a
        if abs(b) <= 1e-6 and abs(a) > 1e-6:
            x_prime = int(-c/a)
            if 0 <= x_prime < width:
                for y_prime in range(height):
                    pixels.append((x_prime, y_prime))
        
        return pixels
 
    
 
    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)
        #import ipdb; ipdb.set_trace()
        
    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            #import ipdb; ipdb.set_trace()
            # loss_contrast = self.loss_terms['contrastive'](self.feats_contrast, torch.tensor(self.contrast_labels, device=self.feats_contrast.device))
        self.backward_term(loss_cos, self.optim)
        update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1, self.master)
        # update_log_term(self.log_terms.get('contrastive'), reduce_tensor(loss_contrast, self.world_size).clone().detach().item(), 1, self.master)

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        # imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
        test_img_length = self.cfg.data.test_length*5
        img_size = self.cfg.size
        results=dict(
            imgs_masks=np.empty((test_img_length, 1, img_size, img_size), dtype='int64'),
            anomaly_maps=np.empty((test_img_length, 1, img_size, img_size), dtype='float32'),
            cls_names=np.empty((test_img_length,), dtype='object'),
            anomalys=np.empty((test_img_length,), dtype='int64'),
            sample_anomaly=np.empty((self.cfg.data.test_length,), dtype='int64'),
            # img_path=np.empty((test_img_length,), dtype='object'),
            # sample_anomaly=np.empty((test_img_length,), dtype='int64')
            # img_path = self.img_path,
            # sample_anomaly = self.sample_anomaly
        )
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            t1 = get_timepc()
            # batch_idx += 1
            #import ipdb; ipdb.set_trace()
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s)
            update_log_term(self.log_terms.get('cos'), reduce_tensor(loss_cos, self.world_size).clone().detach().item(), 1, self.master)
            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s, [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False, amap_mode='add', gaussian_sigma=4)
            anomaly_map = np.expand_dims(anomaly_map, axis=1)
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            # imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            # anomaly_maps.append(anomaly_map)
            # cls_names.append(np.array(self.cls_name))
            # anomalys.append(self.anomaly.cpu().numpy().astype(int))
            # import ipdb; ipdb.set_trace()
            if self.cfg.vis:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
   
            
            if len(test_data['img'].shape) == 5:
                #import ipdb; ipdb.set_trace()
                # 1. not last batch
                if len(test_data['img']) == self.cfg.batch_test_per:
                    results['imgs_masks'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = anomaly_map
                    results['cls_names'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = self.cls_name.flatten()
                    results['anomalys'][5*batch_idx*self.cfg.batch_test_per:5*(batch_idx+1)*self.cfg.batch_test_per] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.sample_anomaly
                # 2. last batch
                else:
                    #import ipdb; ipdb.set_trace()
                    results['imgs_masks'][5*batch_idx*self.cfg.batch_test_per:] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][5*batch_idx*self.cfg.batch_test_per:] = anomaly_map
                    results['cls_names'][5*batch_idx*self.cfg.batch_test_per:] = self.cls_name.flatten()
                    results['anomalys'][5*batch_idx*self.cfg.batch_test_per:] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:] = self.sample_anomaly
            
            else:
                # 1. not last batch
                if len(test_data['img']) == self.cfg.batch_test_per:
                    results['imgs_masks'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = anomaly_map
                    results['cls_names'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.cls_name.flatten()
                    results['anomalys'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:(batch_idx+1)*self.cfg.batch_test_per] = self.sample_anomaly
                # 2. last batch
                else:
                    # import ipdb; ipdb.set_trace()
                    results['imgs_masks'][batch_idx*self.cfg.batch_test_per:] = self.imgs_mask.cpu().numpy().astype(int)
                    results['anomaly_maps'][batch_idx*self.cfg.batch_test_per:] = anomaly_map
                    results['cls_names'][batch_idx*self.cfg.batch_test_per:] = self.cls_name.flatten()
                    results['anomalys'][batch_idx*self.cfg.batch_test_per:] = self.anomaly.flatten().cpu().numpy().astype(int)
                    results['sample_anomaly'][batch_idx*self.cfg.batch_test_per:] = self.sample_anomaly
            
            batch_idx += 1
   
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            # ---------- log ----------
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                    log_msg(self.logger, msg)
        # # merge results
        # if self.cfg.dist:
        # 	results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        # 	torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
        # 	if self.master:
        # 		results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
        # 		valid_results = False
        # 		while not valid_results:
        # 			results_files = glob.glob(f'{self.tmp_dir}/*.pth')
        # 			if len(results_files) != self.cfg.world_size:
        # 				time.sleep(1)
        # 			else:
        # 				idx_result = 0
        # 				while idx_result < self.cfg.world_size:
        # 					results_file = results_files[idx_result]
        # 					try:
        # 						result = torch.load(results_file)
        # 						for k, v in result.items():
        # 							results[k].extend(v)
        # 						idx_result += 1
        # 					except:
        # 						time.sleep(1)
        # 				valid_results = True
        # else:
        # 	results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
        if self.master:
            # results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            msg_csv = copy.deepcopy(msg)
            for key in list(msg_csv.keys()):
                if "(Max)" in key:
                    msg_csv.pop(key)
            msg_csv = tabulate.tabulate(msg_csv, headers='keys', tablefmt=tabulate.simple_separated_format(","), floatfmt='.3f', numalign="center", stralign="center", )
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", )
            log_msg(self.logger, f'\n{msg}')
            #import ipdb; ipdb.set_trace()
            # df = pd.read_html(msg, index_col=0)[0]
            # df.to_csv(f'{self.cfg.logdir}/result.csv')
            with open(f'{self.cfg.logdir}/result.csv', 'w', newline='') as f:
                f.write(msg_csv)
            
