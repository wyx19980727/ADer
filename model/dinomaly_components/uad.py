import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math
import copy

from .dinov3.dinov3.layers import RopePositionEmbedding

from .vision_transformer import bMlp

class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''

    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c // 2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c // 4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c // 2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c // 2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )

    def forward(self, x):
        return self.proj(x)

class MultiProjectionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(384, 384)
        self.proj_b = ProjLayer(384, 384)
        self.proj_c = ProjLayer(384, 384)
        self.proj_d = ProjLayer(384, 384)
        # self.proj_a = bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)
        # self.proj_b = bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)
        # self.proj_c = bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)
        # self.proj_d = bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)

    def forward(self, features):
        return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2]), self.proj_d(features[3])]

class ViTill(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTill, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size
        
        # self.decoder_rope_embed = RopePositionEmbedding(
        #     embed_dim=384,
        #     num_heads=6,
        #     base=100.0,
        #     normalize_coords="separate",
        #     dtype=torch.float32,
        # )

        self.proj_layer = MultiProjectionLayer(embed_dim=384)

        # ''''''
        # # 0. prepare
        # self.num_views, H_img, W_img = 5, 224, 224
        # self.C_feat, self.H_feat, self.W_feat = 384, 16, 16
        # # self.C_feat, self.H_feat, self.W_feat = 256, 16, 16
        # self.patch_size_h, self.patch_size_w = H_img // self.H_feat, W_img // self.W_feat
        # self.num_query = self.H_feat * self.W_feat
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # 1.1 extract query coordinates
        # y_centers = torch.arange(self.H_feat, device=device) * self.patch_size_h + self.patch_size_h / 2 - 0.5  # [H_feat]
        # x_centers = torch.arange(self.W_feat, device=device) * self.patch_size_w + self.patch_size_w / 2 - 0.5  # [W_feat]
        # v_coord, u_coord = torch.meshgrid(y_centers, x_centers, indexing='ij')  # [H_feat, W_feat]，分别是y和x
        # query_points = torch.stack([u_coord, v_coord], dim=0).unsqueeze(0)  # [1, 2, H_feat, W_feat]

        # # 1.2 homogeneous coords of query points
        # x1 = query_points[0, 0].reshape(-1)  # [num_query]
        # y1 = query_points[0, 1].reshape(-1)  # [num_query]
        # ones = torch.ones_like(x1)
        # self.pts1 = torch.stack([x1, y1, ones], dim=1)  # [num_query, 3]
        # self.v2_coord, self.u2_coord = copy.deepcopy(v_coord).reshape(-1), copy.deepcopy(u_coord).reshape(-1)  # both [H_feat*W_feat]

        # # 2. prepare index_pairs
        # class_indices = torch.arange(0, self.num_views, device=device) # exclude top-view
        # index_combinations = torch.cartesian_prod(class_indices, class_indices) # Generate all possible pairs (including self-pairs and ordered pairs)
        # mask = index_combinations[:, 0] != index_combinations[:, 1] # Create a mask to filter out pairs where i == j
        # self.index_combinations = index_combinations[mask]
        
        # # 3. load saved fundamental_matrix
        # path = './matchanything_FM_RANSAC_fundamental_matrix_results_full.pth'
        # self.fundamental_matrix_results = torch.load(path)
        # ''''''

    def forward(self, x, img_path=None):
        # x = self.encoder.prepare_tokens(x)
        # x = self.encoder.patch_embed(x)
        # x, rot_pos_embed = self.encoder._pos_embed(x)
        # x = self.encoder.norm_pre(x)

        # x, hw_tuple = self.encoder.prepare_tokens_with_masks(x)


        # en_list = []
        # for i, blk in enumerate(self.encoder.blocks):
        #     if i <= self.target_layers[-1]:
        #         if i in self.encoder_require_grad_layer:
        #             x = blk(x)
        #         else:
        #             with torch.no_grad():
        #                 x = blk(x)
        #                 # rope_sincos = self.encoder.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
        #                 # x = blk(x, rope_sincos)
        #     else:
        #         continue
        #     if i in self.target_layers:
        #         en_list.append(x)

        # import ipdb; ipdb.set_trace()
        en_list = self.encoder.get_intermediate_layers(x, n=self.target_layers, norm=False)
        #side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
        side = int(math.sqrt(en_list[0].shape[1]))

        en_proj_list = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en_list]
        en_proj_list = self.proj_layer(en_proj_list)
        # en_proj_list = [e.reshape([x.shape[0], -1, side * side]).permute(0, 2, 1).contiguous() for e in en_proj_list]
        # import ipdb; ipdb.set_trace()
        glo_feats = [F.adaptive_avg_pool2d(en_feat, 1).squeeze() for en_feat in en_proj_list]


        # if self.remove_class_token:
        #     en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        # x = self.fuse_feature(en_list)
        en_proj_list = [e.reshape([x.shape[0], -1, side * side]).permute(0, 2, 1).contiguous() for e in en_proj_list]
        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        # glo_feats = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze()

        # en_proj_list = self.proj_layer(en_list)

        # x = self.fuse_feature(en_proj_list)

        # glo_feats = [F.adaptive_avg_pool1d(en_feat.permute(0, 2, 1).contiguous(), 1).squeeze() for en_feat in en_proj_list]
            
        # import ipdb; ipdb.set_trace()
        
        
        
        
        
        # import ipdb; ipdb.set_trace()

        # B = x.shape[0] // self.num_views

        # mid = x[:, 1 + self.encoder.num_register_tokens:, :]
        # mid = mid.permute(0, 2, 1).reshape(B, self.num_views, -1, side, side)  # [B*num_views, C, H_feat, W_feat]

        # '''
        # Correct AVG Epipolar Attention Module
        # '''
        
        # # 2. Enhance by Epipolar Attention (批量并行版本)
        # enhanced_feature = torch.zeros_like(mid)  # 初始化累加器
        # count_matrix = torch.zeros(B, self.num_views, 1, 1, 1, device=mid.device)  # 计数矩阵

        # # 批量计算所有有效基础矩阵
        # batch_fundamental_matrix = torch.zeros(B, len(self.index_combinations), 3, 3, device=mid.device)
        # valid_mask = torch.zeros(B, len(self.index_combinations), dtype=torch.bool, device=mid.device)
        
        # for b in range(B):
        #     for idx, (i, j) in enumerate(self.index_combinations):
        #         img_i_name = img_path[b, i].split('/')[-1]
        #         img_j_name = img_path[b, j].split('/')[-1]
        #         key = f"{img_i_name}_and_{img_j_name}"
        #         if self.fundamental_matrix_results[key] is not None:
        #             batch_fundamental_matrix[b, idx] = self.fundamental_matrix_results[key]
        #             valid_mask[b, idx] = True
                    
        # #import ipdb; ipdb.set_trace()
        # # 仅处理有效矩阵对
        # valid_indices = torch.nonzero(valid_mask)
        # b_ids, pair_ids = valid_indices[:, 0], valid_indices[:, 1]
        # i_src = self.index_combinations[pair_ids, 0]
        # i_tgt = self.index_combinations[pair_ids, 1]
        # F_valid = batch_fundamental_matrix[valid_mask]  # [num_valid, 3, 3]

        # # 批量计算极线约束
        # pts1_expanded = self.pts1.unsqueeze(0).expand(len(F_valid), -1, -1)  # [num_valid, num_query, 3]
        # l = torch.bmm(F_valid, pts1_expanded.transpose(1,2)).transpose(1,2)  # [num_valid, num_query, 3]
        # a, b, c = l.unbind(dim=-1)  
        # a, b, c = a.unsqueeze(2), b.unsqueeze(2), c.unsqueeze(2) # 各[num_valid, num_query, 1]
        
        # # 计算极线距离
        # # eps = 1e-6  # 避免分母为0
        # # denominator = (a**2 + b**2).clamp(min=eps).sqrt()  # 先clamp再开方，确保分母≥sqrt(eps)
        # # dist = torch.abs(a * self.u2_coord + b * self.v2_coord + c) / denominator
        # dist = torch.abs(a * self.u2_coord + b * self.v2_coord + c) / (a**2 + b**2).sqrt() # [num_valid, num_query, H*W]
        # masks = dist < max(self.patch_size_h, self.patch_size_w) / 2  # [num_valid, num_query, H, W]

        # # 批量注意力计算
        # Q = mid[b_ids, i_src].flatten(2).permute(0,2,1)  # [num_valid, num_query, C]
        # K = mid[b_ids, i_tgt].flatten(2).permute(0,2,1)  # [num_valid, num_query, C]
        
        # # 带掩码的注意力机制
        # attn_out = torch.zeros_like(Q)
        # for idx in range(len(F_valid)):
        #     attn_mask = ~masks[idx]  # [num_query, H*W]
        #     attn_out[idx] = F.scaled_dot_product_attention(
        #         Q[idx].unsqueeze(1), 
        #         K[idx].unsqueeze(0).expand(self.num_query, -1, -1),
        #         K[idx].unsqueeze(0).expand(self.num_query, -1, -1),
        #         attn_mask=attn_mask.unsqueeze(1)
        #     ).squeeze(1)
        #     # # NAN check
        #     # if torch.isnan(attn_out[idx]).any():
        #     #     attn_out[idx] = Q[idx]

        # # attn_mask = ~masks
        # # attn_out = F.scaled_dot_product_attention(
        # #     Q.unsqueeze(2), 
        # #     K.unsqueeze(1).expand(-1, self.num_query, -1, -1),
        # #     K.unsqueeze(1).expand(-1, self.num_query, -1, -1),
        # #     attn_mask=attn_mask.unsqueeze(2)
        # # ).squeeze(2)
        
        # # 累加增强特征
        # attn_out = attn_out.permute(0,2,1).view(-1, self.C_feat, self.H_feat, self.W_feat)
        # enhanced_feature[b_ids, i_src] += attn_out
        # count_matrix[b_ids, i_src] += 1

        # # 归一化处理
        # enhanced_feature /= count_matrix.clamp(min=1)  # 避免除零
        # enhanced_feature = mid + enhanced_feature  # 残差连接
        
        # '''
        # Correct AVG Epipolar Attention Module
        # '''
        # x = torch.cat([x[:, :1 + self.encoder.num_register_tokens, :], enhanced_feature.view(B * self.num_views, self.C_feat, -1).permute(0, 2, 1)], dim=1)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            # x = blk(x, attn_mask=attn_mask)
            # rope_sincos = self.decoder_rope_embed(H=hw_tuple[0], W=hw_tuple[1])
            # x = blk(x, rope_sincos)
            x = blk(x)
            # rope_sincos = self.encoder.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
            # x = blk(x, rope_sincos)
            
            de_list.append(x)
        
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        en_proj = [self.fuse_feature([en_proj_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        # en = [en_list[idx] for idxs in self.fuse_layer_encoder for idx in idxs]
        # de = [de_list[idx] for idxs in self.fuse_layer_decoder for idx in idxs]
        # en = [self.fuse_feature2([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        # de = [self.fuse_feature2([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        # import ipdb; ipdb.set_trace()

        # if not self.remove_class_token:  # class tokens have not been removed above
        #     en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
        #     de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]
        
        # if not self.remove_class_token:  # class tokens have not been removed above
        #     en = [e[:, 5:, :] for e in en]
        #     de = [d[:, 5:, :] for d in de]
        
        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        en_proj = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en_proj]

        # glo_feats = [F.adaptive_avg_pool2d(de_feat, 1).squeeze() for de_feat in de]

        return en, de, en_proj, glo_feats

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)
    def fuse_feature2(self, feat_list):
        return torch.cat(feat_list, dim=2)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class ViTillCat(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[1, 3, 5, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(ViTillCat, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        for i, blk in enumerate(self.decoder):
            x = blk(x)

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]
        de = [x]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class ViTAD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 5, 8, 11],
            fuse_layer_encoder=[0, 1, 2],
            fuse_layer_decoder=[2, 5, 8],
            mask_neighbor_size=0,
            remove_class_token=False,
    ) -> None:
        super(ViTAD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de


class ViTillv2(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7]
    ) -> None:
        super(ViTillv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en.append(x)

        x = self.fuse_feature(en)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        side = int(math.sqrt(x.shape[1]))

        en = [e[:, self.encoder.num_register_tokens + 1:, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class ViTillv3(nn.Module):
    def __init__(
            self,
            teacher,
            student,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_dropout=0.,
    ) -> None:
        super(ViTillv3, self).__init__()
        self.teacher = teacher
        self.student = student
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)
        else:
            self.fuse_dropout = nn.Identity()
        self.target_layers = target_layers
        if not hasattr(self.teacher, 'num_register_tokens'):
            self.teacher.num_register_tokens = 0

    def forward(self, x):
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)
            x = patch
            en = []
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)
            en = self.fuse_feature(en, fuse_dropout=False)

        x = patch
        de = []
        for i, blk in enumerate(self.student):
            x = blk(x)
            if i in self.target_layers:
                de.append(x)
        de = self.fuse_feature(de, fuse_dropout=False)

        en = en[:, 1 + self.teacher.num_register_tokens:, :]
        de = de[:, 1 + self.teacher.num_register_tokens:, :]
        side = int(math.sqrt(en.shape[1]))

        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]

    def fuse_feature(self, feat_list, fuse_dropout=False):
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)
            feat = self.fuse_dropout(feat).mean(dim=1)
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
