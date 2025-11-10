import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from . import LOSS
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

@LOSS.register_module
class L1Loss(nn.Module):
    def __init__(self, lam=1):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += self.loss(in1, in2) * self.lam
        return loss


@LOSS.register_module
class L2Loss(nn.Module):
    def __init__(self, lam=1):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += self.loss(in1, in2) * self.lam
        return loss


@LOSS.register_module
class CosLoss(nn.Module):
    def __init__(self, avg=True, flat=True, lam=1):
        super(CosLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity()
        self.lam = lam
        self.avg = avg
        self.flat = flat

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            if self.flat:
                loss += (1 - self.cos_sim(in1.contiguous().view(in1.shape[0], -1), in2.contiguous().view(in2.shape[0], -1))).mean() * self.lam
            else:
                loss += (1 - self.cos_sim(in1.contiguous(), in2.contiguous())).mean() * self.lam
        return loss / len(input1) if self.avg else loss


@LOSS.register_module
class KLLoss(nn.Module):
    def __init__(self, lam=1):
        super(KLLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        # real, pred
        # teacher, student
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            in1 = in1.permute(0, 2, 3, 1)
            in2 = in2.permute(0, 2, 3, 1)
            loss += self.loss(F.log_softmax(in2, dim=-1), F.softmax(in1, dim=-1)) * self.lam
        return loss


@LOSS.register_module
class LPIPSLoss(nn.Module):
    def __init__(self, lam=1):
        super(LPIPSLoss, self).__init__()
        self.loss = None
        self.lam = lam

    def forward(self, input1, input2):
        pass


@LOSS.register_module
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True, lam=1):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
        self.lam = lam

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        loss *= self.lam
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        l = max_val - min_val
    else:
        l = val_range

    padd = window_size//2
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    c1 = (0.01 * l) ** 2
    c2 = (0.03 * l) ** 2

    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret, ssim_map


@LOSS.register_module
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None, lam=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size).cuda()

        self.lam = lam
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        s_score, ssim_map = ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)
        loss = (1.0 - s_score) * self.lam
        return loss

@LOSS.register_module
class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, input):
        loss = torch.fft.fft2(input).abs().mean()
        return loss


@LOSS.register_module
class SumLoss(nn.Module):
    def __init__(self, lam=1):
        super(SumLoss, self).__init__()
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += (in1 + in2) * self.lam
        return loss


@LOSS.register_module
class CSUMLoss(nn.Module):
    def __init__(self, lam=1):
        super(CSUMLoss, self).__init__()
        self.lam = lam

    def forward(self, input):
        loss = 0
        for instance in input:
            _, _, h, w = instance.shape
            loss += torch.sum(instance) / (h * w) * self.lam
        return loss

@LOSS.register_module
class FFocalLoss(nn.Module):
    def __init__(self, lam=1, alpha=-1, gamma=4, reduction="mean"):
        super(FFocalLoss, self).__init__()
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.float()
        targets = targets.float()
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) **  self.gamma)

        if  self.alpha >= 0:
            alpha_t =  self.alpha * targets + (1 -  self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if  self.reduction == "mean":
            loss = loss.mean() * self.lam
        elif  self.reduction == "sum":
            loss = loss.sum() * self.lam

        return loss

@LOSS.register_module
class SegmentCELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, mask, pred):
        bsz,_,h,w=pred.size()
        pred = pred.view(bsz, 2, -1)
        mask = mask.view(bsz, -1).long()
        return self.criterion(pred,mask)
    
@LOSS.register_module
class SupervisedContrastiveLoss(nn.Module):
    """`Supervised Contrastive LOSS <https://arxiv.org/abs/2004.11362>`_.

    This part of code is modified from https://github.com/MegviiDetection/FSCE.

    Args:
        temperature (float): A constant to be divided by consine similarity
            to enlarge the magnitude. Default: 0.2.
        iou_threshold (float): Consider proposals with higher credibility
            to increase consistency. Default: 0.5.
        reweight_type (str): Reweight function for contrastive loss.
            Options are ('none', 'exp', 'linear'). Default: 'none'.
        reduction (str): The method used to reduce the loss into
            a scalar. Default: 'mean'. Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss. Default: 1.0.
    """

    # def __init__(self, temperature = 0.1, loss_weight = 1.0):
    #     # import ipdb; ipdb.set_trace()
    #     super().__init__()
    #     assert temperature > 0, 'temperature should be a positive number.'
    #     self.temperature = temperature
    #     self.loss_weight = loss_weight

    # def forward(self,
    #             features,
    #             labels):
    #     """Forward function.

    #     Args:
    #         features (tensor): Shape of (N, K) where N is the number
    #             of features to be compared and K is the channels.
    #         labels (tensor): Shape of (N).

    #     Returns:
    #         Tensor: The calculated loss.
    #     """
    #     #import ipdb; ipdb.set_trace()
    #     loss_weight = self.loss_weight

    #     assert features.shape[0] == labels.shape[0]

    #     if len(labels.shape) == 1:
    #         labels = labels.reshape(-1, 1)

    #     # mask with shape [N, N], mask_{i, j}=1
    #     # if sample i and sample j have the same label
    #     label_mask = torch.eq(labels, labels.T).float().to(features.device)

    #     similarity = torch.div(
    #         torch.matmul(features, features.T), self.temperature)
    #     # for numerical stability
    #     sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
    #     similarity = similarity - sim_row_max.detach()

    #     # mask out self-contrastive
    #     logits_mask = torch.ones_like(similarity)
    #     logits_mask.fill_diagonal_(0)

    #     exp_sim = torch.exp(similarity) * logits_mask
    #     log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

    #     per_label_log_prob = (log_prob * logits_mask *
    #                           label_mask).sum(1) / label_mask.sum(1)

    #     loss = -per_label_log_prob
    #     loss = loss_weight * loss

    #     return loss.mean()
    def __init__(self, lam=1, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.loss = self.scl_loss
        self.lam = lam
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def scl_loss(self, feats, labels):
        #import ipdb; ipdb.set_trace()
        feats = F.normalize(feats, dim=1)
        
        labels = labels.view(-1, 1)
        
        sim_matrix = torch.mm(feats, feats.T) / self.temperature
        
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)
        scl_loss = -torch.sum(pos_mask * torch.log(torch.exp(sim_matrix) / torch.exp(sim_matrix).sum(dim=-1, keepdim=True))) / pos_mask.sum()
        
        return scl_loss

    def forward(self, x, labels):
        # return self.loss(x, labels) * self.lam
        # import ipdb; ipdb.set_trace()
        loss = [self.scl_loss(glo_feat, labels) for glo_feat in x]
        loss = sum(loss) / len(loss)

        # loss = self.scl_loss(x, labels)
        return loss * self.lam
    


@LOSS.register_module
class LocalContrastiveLoss(nn.Module):
    
    def __init__(self, lam=1, temperature=0.1):
        super(LocalContrastiveLoss, self).__init__()
        self.loss = self.scl_loss
        self.lam = lam
        self.temperature = temperature

    def scl_loss(self, feats_t, feats_t_grid, feats_s, labels, img_path):
        num_views = 5
        
        # Normalize features and Flatten the spatial dimensions
        feats_t = F.normalize(feats_t, p=2, dim=1) # (B, C, H, W)
        feats_t_grid = F.normalize(feats_t_grid, p=2, dim=1)
        
        feats_t_flat = feats_t.view(feats_t.size(0)//5, 5, feats_t.size(1), -1) # (B, V, C, H*W)
        feats_t_grid_flat = feats_t_grid.view(feats_t_grid.size(0)//5, 5, feats_t_grid.size(1), -1) # (B, V, C, H*W)
        
        # feats_s = F.normalize(feats_s, p=2, dim=1)
        # feats_s_flat = feats_s.view(feats_s.size(0)//5, 5, feats_s.size(1), -1)
        
        # recon_sim = F.cosine_similarity(feats_t_grid_flat, feats_s_flat, dim=2)
        # recon_sim = recon_sim / self.temperature  # Apply temperature scaling
        # recon_sim = recon_sim.flatten(start_dim=0, end_dim=1) # (total reconpos_num, H*W)
        
        ''''''
        index_combinations = torch.combinations(torch.arange(num_views), r=2)
        # anchor
        # index_combinations = torch.tensor(((0,1),(0,2),(0,3),(0,4)), device=feats_t.device)
        # cycle
        # index_combinations = torch.tensor(((0,1),(1,2),(2,3),(3,4),(4,0)), device=feats_t.device)
        
        
        # generate postive pairs
        # class_indices = torch.arange(num_views)
        # index_combinations = torch.cartesian_prod(class_indices, class_indices) # Generate all possible pairs (including self-pairs and ordered pairs)
        # mask = index_combinations[:, 0] != index_combinations[:, 1] # Create a mask to filter out pairs where i == j
        # index_combinations = index_combinations[mask]
        # import ipdb; ipdb.set_trace()
        
        q_feats_flat = feats_t_flat[:, index_combinations[:, 0]] # (B, 10, C, H*W)
        k_feats_flat = feats_t_flat[:, index_combinations[:, 1]] # (B, 10, C, H*W)
        
        q_feats_grid_flat = feats_t_grid_flat[:, index_combinations[:, 0]] # (B, 10, C, H*W)
        k_feats_grid_flat = feats_t_grid_flat[:, index_combinations[:, 1]] # (B, 10, C, H*W)
        
        similarity_matrix = torch.einsum('bpci,bpcj->bpij', q_feats_flat, k_feats_flat) # (B, 10, H*W, H*W)
        max_sim_idx = torch.argmax(similarity_matrix, dim=-1) # (B, 10, H*W)
        # random
        #max_sim_idx = torch.randint(0, feats_t_flat.size(-1), size=similarity_matrix.shape[:-1], device=similarity_matrix.device) # Generate random indices
        # # 计算中位数索引 ---------------------------------------------------------
        # _, max_sim_idx = torch.median(
        #         similarity_matrix, 
        #         dim=-1,         # 在最后一个维度计算中位数
        #         keepdim=True    # 保持维度便于后续gather操作
        #     ) # (V, neg_num, H*W, 1)
        # max_sim_idx = max_sim_idx.squeeze(-1)  # 移除多余的维度 (V, neg_num, H*W)
        
        # ''''''''''''
        # #import ipdb; ipdb.set_trace()
        # B, num_pairs, patch_num = max_sim_idx.shape
        # H = W = int(np.sqrt(patch_num)) # Assuming square feature map
        # # Visualize for the first batch and the first pair
        # batch_idx = 2
        # pair_idx = 8
        # current_max_sim_idx = max_sim_idx[batch_idx, pair_idx].reshape(H, W).cpu().numpy()

        # original_image = Image.open(img_path[batch_idx, index_combinations[pair_idx][1]]).convert('RGB')
        # original_image = np.array(original_image)
        # compare_image = Image.open(img_path[batch_idx, index_combinations[pair_idx][0]]).convert('RGB')
        # compare_image = np.array(compare_image)

        # img_h, img_w, _ = original_image.shape
        # patch_h = img_h // H
        # patch_w = img_w // W

        # vis_original_image = original_image.copy()
        # vis_compare_image = compare_image.copy()

        # # Generate a colormap for patches
        # colors = plt.cm.jet(np.linspace(0, 1, H * W))
        # alpha = 0.5  # Transparency for the patches

        # count = -1
        # for i in range(H):
        #     for j in range(W):

        #         count += 1
                
        #         target_patch_index = current_max_sim_idx[i, j]

        #         # if target_patch_index == count:
        #         #     continue

        #         #import ipdb; ipdb.set_trace()

        #         target_patch_row = target_patch_index // W
        #         target_patch_col = target_patch_index % W

        #         color_oirginal = (colors[(i * W + j)] * 255).astype(int) # Get RGBA color
        #         color_tuple_rgb_oirginal = color_oirginal[:3]
        #         color_tuple_rgba_oirginal = color_oirginal # Keep alpha channel
        #         color_compare = (colors[(target_patch_row * W + target_patch_col)] * 255).astype(int) # Get RGBA color
        #         color_tuple_rgb_compare = color_compare[:3]
        #         color_tuple_rgba_compare = color_compare # Keep alpha channel
                

        #         # Source patch bounding box
        #         start_row_q = i * patch_h
        #         start_col_q = j * patch_w
        #         end_row_q = (i + 1) * patch_h
        #         end_col_q = (j + 1) * patch_w

        #         # Target patch bounding box based on max_sim_idx
        #         # start_row_k = target_patch_row * patch_h
        #         # start_col_k = target_patch_col * patch_w
        #         # end_row_k = (target_patch_row + 1) * patch_h
        #         # end_col_k = (target_patch_col + 1) * patch_w

        #         # Color the source and target patches in the visualization with transparency
        #         vis_original_patch = vis_original_image[start_row_q:end_row_q, start_col_q:end_col_q].copy()
        #         vis_compare_patch = vis_compare_image[start_row_q:end_row_q, start_col_q:end_col_q].copy()

        #         colored_patch_oirginal = np.array(color_tuple_rgb_oirginal).reshape(1, 1, 3)
        #         colored_patch_rgba_oirginal = np.array(color_tuple_rgba_oirginal).reshape(1, 1, 4)
        #         colored_patch_compare = np.array(color_tuple_rgb_compare).reshape(1, 1, 3)
        #         colored_patch_rgba_compare = np.array(color_tuple_rgba_compare).reshape(1, 1, 4)

        #         vis_original_image[start_row_q:end_row_q, start_col_q:end_col_q] = (1 - alpha) * vis_original_patch + alpha * colored_patch_oirginal
        #         vis_compare_image[start_row_q:end_row_q, start_col_q:end_col_q] = (1 - alpha) * vis_compare_patch + alpha * colored_patch_compare
        #         # Optionally color target patch as well, or use a different color/overlay for target
        #         # visualization_image[start_row_k:end_row_k, start_col_k:end_col_k] = color_tuple

        #         # Draw rectangle outlines (optional, for better visualization)
        #         # import cv2
        #         # cv2.rectangle(visualization_image, (start_col_q, start_row_q), (end_col_q, end_row_q), color_tuple, 1)
        #         # cv2.rectangle(visualization_image, (start_col_k, start_row_k), (end_col_k, end_row_k), color_tuple, 1)


        # vis_original_image_pil = Image.fromarray(vis_original_image)
        # vis_compare_image_pil = Image.fromarray(vis_compare_image)
        # draw_original = ImageDraw.Draw(vis_original_image_pil)
        # draw_compare = ImageDraw.Draw(vis_compare_image_pil)

        # # # --- Font Size Setting ---
        # # font_size = 16  # Choose your desired font size
        # # try:
        # #     # Try to load a TrueType font (e.g., Arial) - you might need to adjust the path
        # #     font = ImageFont.truetype("~/Arial.ttf", font_size) # or "arialbd.ttf" for bold, etc.
        # # except IOError:
        # #     # If TrueType font is not found, fall back to a default font
        # #     font = ImageFont.load_default()
        # # # --- End Font Size Setting ---
        # font_size = 8
        # font = ImageFont.truetype("/home/albus/Arial.ttf", font_size)


        # count = -1
        # for i in range(H):
        #     for j in range(W):
        #         count += 1
        #         #import ipdb; ipdb.set_trace()
        #         patch_index_str = str(count) # Convert index to string once here
        #         target_patch_index = current_max_sim_idx[i, j]
        #         target_patch_index_str = str(target_patch_index) # Convert target index to string

        #         # if target_patch_index == count:
        #         #     continue
                
        #         #import ipdb; ipdb.set_trace()
        #         # Source patch bounding box
        #         start_row_q = i * patch_h
        #         start_col_q = j * patch_w
        #         end_row_q = (i + 1) * patch_h
        #         end_col_q = (j + 1) * patch_w

        #         # Target patch bounding box based on max_sim_idx
        #         # target_patch_row = target_patch_index // W
        #         # target_patch_col = target_patch_index % W
        #         # start_row_k = target_patch_row * patch_h
        #         # start_col_k = target_patch_col * patch_w
        #         # end_row_k = (target_patch_row + 1) * patch_h
        #         # end_col_k = (target_patch_col + 1) * patch_w

        #         # --- Calculate center positions ---
        #         center_x_original = (start_col_q + end_col_q) // 2
        #         center_y_original = (start_row_q + end_row_q) // 2
        #         # center_x_compare = (start_col_k + end_col_k) // 2
        #         # center_y_compare = (start_row_k + end_row_k) // 2

        #         # --- Get text size ---
        #         text_width_original = draw_original.textlength(patch_index_str, font=font)
        #         text_width_compare = draw_compare.textlength(target_patch_index_str, font=font)

        #         # --- Calculate text position to center ---
        #         text_position_original = (center_x_original - text_width_original // 2, center_y_original - text_width_original // 2)
        #         text_position_compare = (center_x_original - text_width_compare // 2, center_y_original - text_width_compare // 2)


        #         draw_original.text(text_position_original, patch_index_str, font=font, fill=(255,255,255)) # White text
        #         draw_compare.text(text_position_compare, target_patch_index_str, font=font, fill=(255,255,255)) # White text


        # plt.figure(figsize=(10, 10))
        # plt.subplot(1, 2, 1)
        # plt.title("Original Image Patches")
        # plt.axis('off')
        # plt.imshow(np.array(vis_original_image_pil)) # Convert back to numpy for plt.imshow
        # plt.subplot(1, 2, 2)
        # plt.title("Compared Image Patches")
        # plt.axis('off')
        # plt.imshow(np.array(vis_compare_image_pil)) # Convert back to numpy for plt.imshow
        # plt.savefig('vis_patch_correspondence.png', dpi=600) # Save figure with both images
        # plt.close() # Close the plot to prevent display in notebook if running in one
        # # vis_original_image_pil.save('vis_original_image.png') # Save using PIL directly
        # # vis_compare_image_pil.save('vis_compare_image.png') # Save using PIL directly


        # import ipdb; ipdb.set_trace()
        # ''''''''''''

        # Gather k_feat_grid_flat based on max_sim_idx
        gathered_k_feats_grid_flat = torch.gather(k_feats_grid_flat, 3, max_sim_idx.unsqueeze(2).expand(q_feats_grid_flat.size(0), q_feats_grid_flat.size(1), q_feats_grid_flat.size(2), -1)) # (B, 10, C, H*W)

        del similarity_matrix, max_sim_idx

        pos_sim = F.cosine_similarity(q_feats_grid_flat, gathered_k_feats_grid_flat, dim=2) # (B, 10, H*W)
        pos_sim = pos_sim / self.temperature  # Apply temperature scaling
        pos_sim = pos_sim.flatten(start_dim=0, end_dim=1) # (total pos_num, H*W)
        
        
        # pos_sim = torch.cat([pos_sim, recon_sim], dim=0)
        
        ''''''
        
        # NEGATIVE SIMILARITY
        cls_labels = labels[::5]
        
        neg_sim_list = []
        for i in range(feats_t_flat.size(0)):
            #import ipdb; ipdb.set_trace()
            # Get indices of all different-class samples
            neg_indices = torch.where(cls_labels != cls_labels[i])[0]
           
            neg_k_flat = torch.flatten(feats_t_flat[neg_indices], start_dim=0, end_dim=1) # (neg_num, C, H*W)
            neg_k_grid_flat = torch.flatten(feats_t_grid_flat[neg_indices], start_dim=0, end_dim=1) # (neg_num, C, H*W)
            similarity_matrix = torch.einsum('vci,ncj->vnij', feats_t_flat[i], neg_k_flat) # (V, neg_num, H*W, H*W)
            
            min_sim_idx = torch.argmax(similarity_matrix, dim=-1) # (V, neg_num, H*W)
            #min_sim_idx = torch.randint(0, feats_t_flat.size(-1), size=similarity_matrix.shape[:-1], device=similarity_matrix.device) # Generate random indices
            # 计算中位数索引 ---------------------------------------------------------
            #import ipdb; ipdb.set_trace()
            # _, min_sim_idx = torch.median(
            #     similarity_matrix, 
            #     dim=-1,         # 在最后一个维度计算中位数
            #     keepdim=True    # 保持维度便于后续gather操作
            # ) # (V, neg_num, H*W, 1)
            # min_sim_idx = min_sim_idx.squeeze(-1)  # 移除多余的维度 (V, neg_num, H*W)
            
            
            q_feat_grid_flat = feats_t_grid_flat[i].unsqueeze(1) # (V, 1, C, H*W)

            # Gather neg_k_grid_flat based on min_sim_idx
            gathered_neg_k_grid_flat = torch.gather(neg_k_grid_flat.expand(num_views, -1, -1, -1), 3, min_sim_idx.unsqueeze(2).expand(-1, -1, q_feat_grid_flat.size(2), -1)) # (V, neg_num, C, H*W)
            
            del similarity_matrix, min_sim_idx
            
            neg_sim = F.cosine_similarity(q_feat_grid_flat, gathered_neg_k_grid_flat, dim=2) # (V, neg_num, H*W)
            neg_sim_list.append(neg_sim.flatten(start_dim=0, end_dim=1)) 

        neg_sim = torch.cat(neg_sim_list, dim=0)  # (b, h*w) or less if some batches have no negative samples
        del neg_sim_list
        neg_sim = neg_sim / self.temperature  # Apply temperature scaling # (total neg_num, H*W)
        
        #loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim)) + 1e-6))
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim)) + 1e-6))

        return loss.mean()
        
    def forward(self, feats_t, feats_t_grid, feats_s_grid, labels, img_path):

        loss = [self.scl_loss(feat_t, feat_t_grid, feat_s_grid, labels, img_path) for feat_t, feat_t_grid, feat_s_grid in zip(feats_t, feats_t_grid, feats_s_grid)]
        loss = sum(loss) / len(loss)
        #import ipdb; ipdb.set_trace()
        # loss = self.scl_loss(feats_t[2], feats_t_grid[2], feats_s_grid[2], labels, img_path)
        return loss * self.lam

  
        # # Get indices of all different-class samples
        # neg_indices = torch.where(labels != labels[i].item())[0]
        
        # import ipdb; ipdb.set_trace()
        
        
        
        # feats_
        
        # # POSITIVE SIMILARITY
        # pos_sim_list = []
        
        # # gather the features of the same class
        # label_classes = torch.unique(labels)
        
        # for label in label_classes:
            
        #     label_indices = torch.where(labels == label)[0]
        #     label_feats_t_flat = feats_t_flat[label_indices]
        #     label_feats_t_grid_flat = feats_t_grid_flat[label_indices]
            
        #     index_combinations = torch.combinations(torch.arange(label_feats_t_flat.shape[0]), r=2)
        #     # generate postive pairs
        #     # class_indices = torch.arange(label_feats_t_flat.shape[0])
        #     # index_combinations = torch.cartesian_prod(class_indices, class_indices) # Generate all possible pairs (including self-pairs and ordered pairs)
        #     # mask = index_combinations[:, 0] != index_combinations[:, 1] # Create a mask to filter out pairs where i == j
        #     # index_combinations = index_combinations[mask]
            
            
        #     q_feats_flat = label_feats_t_flat[index_combinations[:, 0]] # (N, C, H*W)
        #     k_feats_flat = label_feats_t_flat[index_combinations[:, 1]] # (N, C, H*W)

        #     q_feats_grid_flat = label_feats_t_grid_flat[index_combinations[:, 0]] # (N, C, H*W)
        #     k_feats_grid_flat = label_feats_t_grid_flat[index_combinations[:, 1]] # (N, C, H*W)

        #     similarity_matrix = torch.einsum('nci,ncj->nij', q_feats_flat, k_feats_flat) # (N, H*W, H*W)
        #     max_sim_idx = torch.argmax(similarity_matrix, dim=-1) # (N, H*W)

        #     # Gather k_feat_grid_flat based on max_sim_idx
        #     gathered_k_feats_grid_flat = torch.gather(k_feats_grid_flat, 2, max_sim_idx.unsqueeze(1).expand(-1, q_feats_grid_flat.size(1), -1)) # (N, C, H*W)

        #     pos_sim = F.cosine_similarity(q_feats_grid_flat, gathered_k_feats_grid_flat, dim=1) # (N, H*W)
        #     pos_sim_list.append(pos_sim)
        
        # pos_sim = torch.cat(pos_sim_list, dim=0)
        # pos_sim = pos_sim / self.temperature  # Apply temperature scaling
        
        # # NEGATIVE SIMILARITY
        # neg_sim_list = []
        # for i in range(feats_t.size(0)):
        #     # Get indices of all different-class samples
        #     neg_indices = torch.where(labels != labels[i].item())[0]

        #     # If no negative samples are available, skip this sample
        #     if len(neg_indices) == 0:
        #         return torch.tensor(0.0, device=feats_t.device)
            
        #     # # Randomly select one negative sample from different-class samples
        #     # rand_idx = torch.randint(0, len(neg_indices), (2,))
        #     # neg_indices = neg_indices[rand_idx]

        #     neg_k_flat = feats_t_flat[neg_indices]  # (neg_num, c, h*w)
        #     neg_k_grid_flat = feats_t_grid_flat[neg_indices]  # (neg_num, c, h*w)

        #     # Vectorized negative similarity calculation
        #     similarity_matrix = torch.einsum('ci,ncj->nij', feats_t_flat[i], neg_k_flat) # (neg_num, H*W, *H*W)
        #     min_sim_idx = torch.argmin(similarity_matrix, dim=-1) # (neg_num, H*W)
        #     q_feat_grid_flat = feats_t_grid_flat[i].unsqueeze(0) # (1, c, h*w)

        #     # Gather neg_k_grid_flat based on min_sim_idx
        #     gathered_neg_k_grid_flat = neg_k_grid_flat.gather(2, min_sim_idx.unsqueeze(1).expand(-1, q_feat_grid_flat.size(1), -1)) # (1, c, h*w)

        #     neg_sim = F.cosine_similarity(q_feat_grid_flat, gathered_neg_k_grid_flat, dim=1) # (1, h*w)
        #     neg_sim_list.append(neg_sim)

        # #import ipdb; ipdb.set_trace()
        # neg_sim = torch.cat(neg_sim_list, dim=0)  # (b, h*w) or less if some batches have no negative samples
        # neg_sim = neg_sim / self.temperature  # Apply temperature scaling

        # loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim)) + 1e-6))

        # return loss.mean()
        
        
        
        
        
        
        
        
        
        
        # 对比损失（InfoNCE）
        # import ipdb; ipdb.set_trace()
        # numerator = torch.exp(pos_sim).sum(dim=0)
        # denominator = numerator + torch.exp(neg_sim).sum(dim=0)
        # loss = -torch.log(numerator / (denominator))

        # loss1 = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        # loss2 = -torch.log(torch.sum(torch.exp(pos_sim), dim=0) / (torch.sum(torch.exp(pos_sim), dim=0) + torch.sum(torch.exp(neg_sim), dim=0)))
        # loss3 = -torch.log(torch.sum(torch.exp(pos_sim)) / (torch.sum(torch.exp(pos_sim)) + torch.sum(torch.exp(neg_sim)) + 1e-6))

  
    
        # # Contrastive loss (InfoNCE Loss)
        # #loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        # #loss1 = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim))))
        # loss = -torch.log(torch.sum(torch.exp(pos_sim)) / torch.sum((torch.exp(pos_sim)) + torch.sum(torch.exp(neg_sim))))
        # loss = -torch.log(torch.sum(torch.exp(pos_sim), dim=0) / (torch.sum(torch.exp(pos_sim), dim=0) + torch.sum(torch.exp(neg_sim), dim=0) + 1e-6))








        
        # # 生成同类样本的掩码矩阵 [1,5](@ref)
        # mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        # diag_mask = ~torch.eye(mask.size(0), dtype=torch.bool, device=mask.device)
        # pos_mask = mask & diag_mask  # 排除自身
        
        
        
        # import ipdb; ipdb.set_trace()
        
        # # 批量计算所有样本对相似度 [3,4](@ref)
        # all_pairs = torch.einsum('bcl,dcm->bdlm', feats_t_flat, feats_t_flat)  # (B, H*W, H*W)
        # max_sim_idx = torch.argmax(all_pairs, dim=-1)  # (B, B)
        
        # q_grid = feats_t_grid_flat.unsqueeze(1).expand(-1, feats_t.size(0), -1, -1)  # (B, B, C, H*W)
        # k_grid = feats_t_grid_flat.unsqueeze(0).expand(feats_t.size(0), -1, -1, -1)
        # pos_sim = F.cosine_similarity(q_grid, k_grid.gather(3, max_sim_idx.unsqueeze(2).expand(-1, -1, q_grid.size(2), -1)), dim=2)
        
        
        

        # # Normalize features
        # feats_t = F.normalize(feats_t, p=2, dim=1)
        # feats_t_grid = F.normalize(feats_t_grid, p=2, dim=1)
        
        # ''''''
        # # feats_s = F.normalize(feats_s, p=2, dim=1)
        # ''''''
        
        # # Flatten the spatial dimensions
        # feats_t_flat = feats_t.view(feats_t.size(0), feats_t.size(1), -1)
        # feats_t_grid_flat = feats_t_grid.view(feats_t_grid.size(0), feats_t_grid.size(1), -1)
        
        # ''''''
        # # feats_s_flat = feats_s.view(feats_s.size(0), feats_s.size(1), -1)
        # # recon_pos_sim = F.cosine_similarity(feats_t_grid_flat, feats_s_flat, dim=1)
        # ''''''
        
        # # POSITIVE SIMILARITY
        # pos_sim_list = []
        
        # # gather the features of the same class
        # label_classes = torch.unique(labels)
            
        # for label in label_classes:
        #     label_indices = torch.where(labels == label)[0]
        #     label_feats_t = feats_t[label_indices]
        #     label_feats_t_grid = feats_t_grid[label_indices]
            
        #     index_combinations = torch.combinations(torch.arange(label_feats_t.shape[0]), r=2)
            
        #     for combination in index_combinations:
        #         q_feat, k_feat = label_feats_t[combination]
        #         q_feat_flat, k_feat_flat = q_feat.view(q_feat.shape[0],-1).unsqueeze(0), k_feat.view(k_feat.shape[0],-1).unsqueeze(0)
                
        #         q_feat_grid, k_feat_grid = label_feats_t_grid[combination]
        #         q_feat_grid_flat, k_feat_grid_flat = q_feat_grid.view(q_feat_grid.shape[0],-1).unsqueeze(0), k_feat_grid.view(k_feat_grid.shape[0],-1).unsqueeze(0)
                
        #         similarity_matrix = torch.einsum('bci,bcj->bij', q_feat_flat, k_feat_flat)
        #         max_sim_idx = torch.argmax(similarity_matrix, dim=-1)
                
        #         pos_sim = F.cosine_similarity(q_feat_grid_flat, k_feat_grid_flat.gather(2, max_sim_idx.unsqueeze(1).expand(-1, q_feat_grid_flat.size(1), -1)), dim=1)
                
        #         pos_sim_list.append(pos_sim)
        
        # pos_sim = torch.cat(pos_sim_list, dim=0)  # (b, h*w)

        # ''''''
   
        # pos_sim = pos_sim / self.temperature  # Apply temperature scaling
        
        # # Prepare to store negative similarities (one negative sample per batch element)
        # neg_sim_list = []
        # for i in range(feats_t.size(0)):
            
        #     # Get indices of all different-class samples
        #     neg_indices = torch.where(labels != labels[i].item())[0]

        #     # If no negative samples are available, skip this sample
        #     if len(neg_indices) == 0:
        #         return torch.tensor(0.0, device=feats_t.device)

        #     # Randomly select one negative sample from different-class samples
        #     rand_idx = torch.randint(0, len(neg_indices), (2,))
        #     neg_k_grid_flat = feats_t_grid_flat[neg_indices[rand_idx]]  # (1, c, h*w)
        #     # neg_k_grid_flat = feats_t_grid_flat[neg_indices]  # (1, c, h*w)
            
        #     ''''''
        #     # similarity_matrix = torch.einsum('bci,bcj->bij', feats_t_flat[i].unsqueeze(0), neg_k_grid_flat)
        #     # min_sim_idx = torch.argmin(similarity_matrix, dim=-1)
        #     # q_feat_flat = feats_t_grid_flat[i].unsqueeze(0)
            
        #     # neg_sim = F.cosine_similarity(q_feat_flat, neg_k_grid_flat.gather(2, min_sim_idx.unsqueeze(1).expand(-1, q_feat_flat.size(1), -1)), dim=1)
        #     ''''''

        #     # Compute the cosine similarity between q_grid_flat[i] and the selected negative sample
        #     neg_sim = F.cosine_similarity(feats_t_grid_flat[i].unsqueeze(0), neg_k_grid_flat, dim=1)  # (1, h*w)

        #     # Store the negative similarity
        #     neg_sim_list.append(neg_sim)

        # # Concatenate all negative similarities
        # neg_sim = torch.cat(neg_sim_list, dim=0)  # (b, h*w)
        # neg_sim = neg_sim / self.temperature  # Apply temperature scaling
        # import ipdb; ipdb.set_trace()
        
        
        # # Contrastive loss (InfoNCE Loss)
        # #loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-6))
        # #loss1 = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim))))
        # loss = -torch.log(torch.sum(torch.exp(pos_sim)) / torch.sum((torch.exp(pos_sim)) + torch.sum(torch.exp(neg_sim))))
        # loss = -torch.log(torch.sum(torch.exp(pos_sim), dim=0) / (torch.sum(torch.exp(pos_sim), dim=0) + torch.sum(torch.exp(neg_sim), dim=0) + 1e-6))
        
        
        
        
        # return loss.mean()

'''  
@LOSS.register_module
class LocalContrastiveLoss(nn.Module):
    
    def __init__(self, lam=1, temperature=0.1):
        super(LocalContrastiveLoss, self).__init__()
        self.loss = self.scl_loss
        self.lam = lam
        self.temperature = temperature

    def scl_loss(self, feats_t, feats_t_grid, feats_s, labels):
        #import ipdb; ipdb.set_trace()

        # Normalize features
        num_views = 5
        feats_t = F.normalize(feats_t, p=2, dim=1)
        feats_t_grid = F.normalize(feats_t_grid, p=2, dim=1)
        
        ''''''
        feats_s = F.normalize(feats_s, p=2, dim=1)
        
        #import ipdb; ipdb.set_trace()
        
        # feats_t = torch.cat([feats_t, feats_s], dim=0)
        # feats_t_grid = torch.cat([feats_t_grid, feats_s],dim=0)
        # feats_all = torch.cat([feats_t, feats_s], dim=0)
        # feats_all_grid = torch.cat([feats_t_grid, feats_s], dim=0)
        
        
        ''''''
        
        # Flatten the spatial dimensions
        feats_t_flat = feats_t.view(feats_t.size(0), feats_t.size(1), -1)
        feats_t_grid_flat = feats_t_grid.view(feats_t_grid.size(0), feats_t_grid.size(1), -1)
        
        ''''''
        feats_s_flat = feats_s.view(feats_s.size(0), feats_s.size(1), -1)
        recon_pos_sim = F.cosine_similarity(feats_t_grid_flat, feats_s_flat, dim=1)



        ''''''
        
        
        # Split the features into num_views parts
        feats_vt = feats_t.view(feats_t.size(0)//num_views, num_views, feats_t.size(1), -1)
        feats_vt_grid = feats_t_grid.view(feats_t_grid.size(0)//num_views, num_views, feats_t_grid.size(1), -1)
        # feats_vs = feats_s.view(feats_s.size(0)//num_views, num_views, feats_s.size(1), -1)
        ''''''
        # feats_vt = feats_all.view(feats_all.size(0)//num_views, num_views, feats_all.size(1), -1)
        # feats_vt_grid = feats_all_grid.view(feats_all_grid.size(0)//num_views, num_views, feats_all_grid.size(1), -1)
        
        ''''''
        pos_sim_list = []
        # pos_sim_s_list = []
        
        # ## start ##
        # feats_ts = torch.chunk(feats_vt, num_views, dim=1)
        # feats_ts_grid = torch.chunk(feats_vt_grid, num_views, dim=1)
        # # feats_ss = torch.chunk(feats_vs, num_views, dim=1)
        # ''''''
        
        # for i in range(len(feats_ts)):
        #     q_feat = feats_ts[i].squeeze(1)
        #     q_feat_grid = feats_ts_grid[i].squeeze(1)
        #     # q_feat_s = feats_ss[i].squeeze(1)
            
        #     for j in range(len(feats_ts)):
        #         if i != j:
        #             k_feat = feats_ts[j].squeeze(1)
        #             k_feat_grid = feats_ts_grid[j].squeeze(1)
        #             # k_feat_s = feats_ss[j].squeeze(1)
                    
        #             similarity_matrix = torch.einsum('bci,bcj->bij', q_feat, k_feat)
        #             max_sim_idx = torch.argmax(similarity_matrix, dim=-1)
                    
        #             pos_sim = F.cosine_similarity(q_feat_grid, k_feat_grid.gather(2, max_sim_idx.unsqueeze(1).expand(-1, q_feat.size(1), -1)), dim=1)
        #             pos_sim_list.append(pos_sim)
                    
        #             # pos_sim_s = F.cosine_similarity(q_feat_s, k_feat_s, dim=1)
        #             # pos_sim_s_list.append(pos_sim_s)
        # pos_sim = torch.cat(pos_sim_list, dim=0)  # (b, h*w)
        # ''''''
        # #import ipdb; ipdb.set_trace() 
        # # pos_sim_s = torch.cat(pos_sim_s_list, dim=0)
        
        # pos_sim = torch.cat([pos_sim, recon_pos_sim], dim=0)
        
        # ## end ##
        
        #import ipdb; ipdb.set_trace()
        
        # gather the features of the same class
        label_classes = torch.unique(labels)
        pos_sim_list = []
        
        
        for label in label_classes:
            #import ipdb; ipdb.set_trace()
            label_indices = torch.where(labels == label)[0]
            label_feats_t = feats_t[label_indices]
            label_feats_t_grid = feats_t_grid[label_indices]
            
            index_combinations = torch.combinations(torch.arange(label_feats_t.shape[0]), r=2)
            
            for combination in index_combinations:
                #import ipdb; ipdb.set_trace()
                q_feat, k_feat = label_feats_t[combination]
                q_feat_flat, k_feat_flat = q_feat.view(q_feat.shape[0],-1).unsqueeze(0), k_feat.view(k_feat.shape[0],-1).unsqueeze(0)
                
                q_feat_grid, k_feat_grid = label_feats_t_grid[combination]
                q_feat_grid_flat, k_feat_grid_flat = q_feat_grid.view(q_feat_grid.shape[0],-1).unsqueeze(0), k_feat_grid.view(k_feat_grid.shape[0],-1).unsqueeze(0)
                
                similarity_matrix = torch.einsum('bci,bcj->bij', q_feat_flat, k_feat_flat)
                max_sim_idx = torch.argmax(similarity_matrix, dim=-1)
                
                pos_sim = F.cosine_similarity(q_feat_grid_flat, k_feat_grid_flat.gather(2, max_sim_idx.unsqueeze(1).expand(-1, q_feat_grid_flat.size(1), -1)), dim=1)
                
                pos_sim_list.append(pos_sim)
        
        pos_sim = torch.cat(pos_sim_list, dim=0)  # (b, h*w)
        pos_sim = torch.cat([pos_sim, recon_pos_sim], dim=0)
        #import ipdb; ipdb.set_trace()

        ''''''
   
        pos_sim = pos_sim / self.temperature  # Apply temperature scaling
        
        # Prepare to store negative similarities (one negative sample per batch element)
        neg_sim_list = []
        
        ''''''
        # labels = torch.cat([labels, labels], dim=0)
        ''''''

        for i in range(feats_t.size(0)):
            
            # Get indices of all different-class samples
            neg_indices = torch.where(labels != labels[i].item())[0]

            # If no negative samples are available, skip this sample
            if len(neg_indices) == 0:
                return torch.tensor(0.0, device=feats_t.device)

            # Randomly select one negative sample from different-class samples
            # rand_idx = torch.randint(0, len(neg_indices), (4,))
            # neg_k_grid_flat = feats_t_grid_flat[neg_indices[rand_idx]]  # (1, c, h*w)
            neg_k_grid_flat = feats_t_grid_flat[neg_indices]  # (1, c, h*w)
            
            ''''''
            similarity_matrix = torch.einsum('bci,bcj->bij', feats_t_flat[i].unsqueeze(0), neg_k_grid_flat)
            min_sim_idx = torch.argmin(similarity_matrix, dim=-1)
            q_feat_flat = feats_t_grid_flat[i].unsqueeze(0)
            
            neg_sim = F.cosine_similarity(q_feat_flat, neg_k_grid_flat.gather(2, min_sim_idx.unsqueeze(1).expand(-1, q_feat_flat.size(1), -1)), dim=1)
            ''''''

            # Compute the cosine similarity between q_grid_flat[i] and the selected negative sample
            # neg_sim = F.cosine_similarity(feats_t_grid_flat[i].unsqueeze(0), neg_k_grid_flat, dim=1)  # (1, h*w)

            # Store the negative similarity
            neg_sim_list.append(neg_sim)

        # Concatenate all negative similarities
        neg_sim = torch.cat(neg_sim_list, dim=0)  # (b, h*w)
        neg_sim = neg_sim / self.temperature  # Apply temperature scaling
        #import ipdb; ipdb.set_trace()
        
        
        # Contrastive loss (InfoNCE Loss)
        #loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-6))
        #loss1 = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim))))
        loss = -torch.log(torch.sum(torch.exp(pos_sim)) / torch.sum((torch.exp(pos_sim)) + torch.sum(torch.exp(neg_sim))))
        
        return loss.mean()

    def forward(self, feats_t, feats_t_grid, feats_s_grid, labels):

        loss = [self.scl_loss(feat_t, feat_t_grid, feat_s_grid, labels) for feat_t, feat_t_grid, feat_s_grid in zip(feats_t, feats_t_grid, feats_s_grid)]
        loss = sum(loss) / len(loss)
        return loss * self.lam
'''

    
@LOSS.register_module
class SCLLoss(nn.Module):
    def __init__(self, lam=1, temperature=0.1):
        super(SCLLoss, self).__init__()
        self.loss = self.scl_loss
        self.lam = lam
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def scl_loss(self, feats, feats_grid, labels):
        #import ipdb; ipdb.set_trace()
        feats = torch.cat([feats, feats_grid], dim=0)
        feats = F.normalize(feats, dim=1)
        
        labels = labels.view(-1, 1)
        labels = torch.cat([labels, labels], dim=0)
        
        sim_matrix = torch.mm(feats, feats.T) / self.temperature
        
        pos_mask = (labels == labels.T).float()
        pos_mask.fill_diagonal_(0)
        scl_loss = -torch.sum(pos_mask * torch.log(torch.exp(sim_matrix) / torch.exp(sim_matrix).sum(dim=-1, keepdim=True))) / pos_mask.sum()
        
        return scl_loss

    def forward(self, x, x_k, labels):
        return self.loss(x, x_k, labels) * self.lam


@LOSS.register_module
class DenseLoss(nn.Module):
    def __init__(self, lam=1, temperature=0.1):
        super(DenseLoss, self).__init__()
        self.loss = self.densecl
        self.lam = lam
        self.temperature = temperature

    def densecl(self, q_b, k_b, q_grid, k_grid, labels):
        # Normalize features
        #import ipdb; ipdb.set_trace()
        q_b = F.normalize(q_b, p=2, dim=1)
        k_b = F.normalize(k_b, p=2, dim=1)
        q_grid = F.normalize(q_grid, p=2, dim=1)
        k_grid = F.normalize(k_grid, p=2, dim=1) 

        # Flatten the spatial dimensions
        q_b_flat = q_b.view(q_b.size(0), q_b.size(1), -1) 
        k_b_flat = k_b.view(k_b.size(0), k_b.size(1), -1) 
        similarity_matrix = torch.einsum('bci,bcj->bij', q_b_flat, k_b_flat) 

        # Get the index of the most similar features between q_b and k_b
        max_sim_idx = torch.argmax(similarity_matrix, dim=-1)  # (b, h*w)

        # Flatten q_grid and k_grid for grid-level comparison
        q_grid_flat = q_grid.view(q_grid.size(0), q_grid.size(1), -1)  # (b, c, h*w)
        k_grid_flat = k_grid.view(k_grid.size(0), k_grid.size(1), -1)  # (b, c, h*w)

        # Calculate the positive similarity for the same class
        pos_sim = F.cosine_similarity(
            q_grid_flat,
            k_grid_flat.gather(2, max_sim_idx.unsqueeze(1).expand(-1, q_grid.size(1), -1)),
            dim=1
        )  # (b, h*w)

        # Apply temperature scaling to the positive similarity
        pos_sim = pos_sim / self.temperature

        # Prepare to store negative similarities (one negative sample per batch element)
        neg_sim_list = []

        for i in range(q_b.size(0)):
            # Get indices of all different-class samples
            neg_indices = torch.where(labels != labels[i].item())[0]

            # If no negative samples are available, skip this sample
            if len(neg_indices) == 0:
                return torch.tensor(0.0, device=q_b.device)

            # Randomly select one negative sample from different-class samples
            rand_idx = torch.randint(0, len(neg_indices), (1,))
            neg_k_grid_flat = k_grid_flat[neg_indices[rand_idx]]  # (1, c, h*w)

            # Compute the cosine similarity between q_grid_flat[i] and the selected negative sample
            neg_sim = F.cosine_similarity(q_grid_flat[i].unsqueeze(0), neg_k_grid_flat, dim=1)  # (1, h*w)

            # Store the negative similarity
            neg_sim_list.append(neg_sim)

        # Concatenate all negative similarities
        neg_sim = torch.cat(neg_sim_list, dim=0)  # (b, h*w)
        neg_sim = neg_sim / self.temperature  # Apply temperature scaling

        # Contrastive loss (InfoNCE Loss)
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-6))

        return loss.mean()

    def forward(self, q_b, k_b, q_grid, k_grid, labels):
        #import ipdb; ipdb.set_trace()
        if not isinstance(q_b, list):
            q_b = [q_b]
            k_b = [k_b]
            q_grid = [q_grid]
            k_grid = [k_grid]

        loss = [self.densecl(q, k, q_g, k_g, labels) for q, k, q_g, k_g in zip(q_b, k_b, q_grid, k_grid)]
        loss = sum(loss) / len(loss)
        return loss * self.lam