import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from einops import rearrange, reduce
from timm.models.resnet import Bottleneck
import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from model import get_model
from model import MODEL


# ========== MFCN ==========
class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(MFCN, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [
            in_stride / outstrides[0] for in_stride in instrides
        ]  # for resize
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]

    def forward(self, input):
        # features = input["features"]
        features = input
        assert len(self.inplanes) == len(features)

        feature_list = []
        # resize & concatenate
        for i in range(len(features)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(features[i])
            feature_list.append(feature_resize)

        feature_align = torch.cat(feature_list, dim=1)
        return feature_align
        # return {"feature_align": feature_align, "outplane": self.get_outplanes()}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides


# ========== UniAD ==========
class UniAD_decoder(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        self.save_recon = save_recon

        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input, img_path=None):
        # feature_align = input["feature_align"]  # B x C X H x W
        feature_align = input
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        #
        
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed, img_path=img_path
        )  # (H x W) x B x C
        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        # if not self.training and self.save_recon:
        # 	clsnames = input["clsname"]
        # 	filenames = input["filename"]
        # 	for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
        # 		filedir, filename = os.path.split(filename)
        # 		_, defename = os.path.split(filedir)
        # 		filename_, _ = os.path.splitext(filename)
        # 		save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
        # 		os.makedirs(save_dir, exist_ok=True)
        # 		feature_rec_np = feat_rec.detach().cpu().numpy()
        # 		np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        # return {
        # 	"feature_rec": feature_rec,
        # 	"feature_align": feature_align,
        # 	"pred": pred,
        # }
        return feature_align, feature_rec, pred


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        neighbor_mask,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        # ''''''
        # # 0. prepare
        # self.num_views, H_img, W_img = 5, 256, 256
        # self.C_feat, self.H_feat, self.W_feat = 256, 8, 8
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
        # path = './fundamental_matrix_results_full.pth'
        # self.fundamental_matrix_results = torch.load(path)
        # ''''''

        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, 
            # self.C_feat, self.H_feat, self.W_feat, self.num_views, self.pts1, self.v2_coord, self.u2_coord, self.index_combinations, self.fundamental_matrix_results,
            dim_feedforward, dropout, activation, normalize_before,
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
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
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, src, pos_embed, img_path=None):
        #import ipdb; ipdb.set_trace()
        _, batch_size, _ = src.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder(
            src, mask=mask_enc, pos=pos_embed, img_path=img_path
        )  # (H X W) x B x C
        
        
        # B = output_encoder.shape[1] // self.num_views
        # mid = rearrange(output_encoder, "(h w) (b v) c -> b v c h w", h=self.H_feat, v=self.num_views)

        
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
        
        

        # '''
        # Epipolar Attention Module
        # '''
        
        # for batch in range(B):
        #     # 1. for each pair, extract the corresponding images
        #     for index_pair in self.index_combinations:
        #         # 1.1 prepare the two images
        #         i, j = index_pair
        #         img_i_path = img_path[:, i]
        #         img_j_path = img_path[:, j]
        #         img_i_name = img_i_path[batch].split('/')[-1]
        #         img_j_name = img_j_path[batch].split('/')[-1]
        #         fundamental_key = img_i_name + "_and_" + img_j_name
                
        #         if self.fundamental_matrix_results[fundamental_key] is None:
        #             continue
        #         else:
        #             # 1.1 calculate fundamental matrix
        #             Fundamental_matrix = self.fundamental_matrix_results[fundamental_key]
                    
        #             # 1.2. Epipolar line
        #             # l = F @ pt1.T, 结果 shape [3, num_query]，转置后 [num_query, 3]
        #             l = torch.matmul(Fundamental_matrix, self.pts1.t()).t()  # [num_query, 3]
        #             a = l[:, 0].unsqueeze(1)  # [num_query, 1]
        #             b = l[:, 1].unsqueeze(1)  # [num_query, 1]
        #             c = l[:, 2].unsqueeze(1)  # [num_query, 1]
                    
        #             # 1.3. Epipolar distance
        #             dist = torch.abs(a * self.u2_coord + b * self.v2_coord + c) / (a.pow(2) + b.pow(2)).sqrt()  # [num_query, H_feat*W_feat]
        #             dist = dist.reshape(self.num_query, self.H_feat, self.W_feat)  # [num_query, H_feat, W_feat]
        #             # 距离小于阈值的patch视为在极线上
        #             masks = dist < max(self.patch_size_h, self.patch_size_w) / 2  # [num_query, H_feat, W_feat]
                    
        #             # 1. Extract query embedding and image patches
        #             query_embedding = shortcut[batch, i, :, :].reshape(self.C_feat, self.num_query).transpose(0, 1)   # [num_query, C]
        #             img2_patches = shortcut[batch, j, :, :].reshape(self.C_feat, self.num_query).transpose(0, 1)  # [num_query, C]
        #             masks_flat = masks.reshape(self.num_query, -1)  # [num_query, 1024]
                    
        #             # 2. Construct Q,K,V
        #             Q = query_embedding.unsqueeze(1)  # [num_query, 1, C]
        #             K = img2_patches.unsqueeze(0).expand(self.num_query, -1, -1)  # [num_query, 1024, C]
        #             V = K  # [num_query, 1024, C]
                    
        #             # 3. 对mask为False的位置赋极小值，防止被softmax
        #             attn_mask = ~masks_flat  # [num_query, 1024], True为需要mask掉
        #             # scaled_dot_product_attention 支持 attn_mask: [num_query, 1, 1024]
        #             attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask.unsqueeze(1))  # [num_query, 1, C]
        #             attn_out = attn_out.squeeze(1)  # [num_query, C]
                    
        #             # # 若某query没有极线patch（全mask），其attn_out为nan，可用原特征替换
        #             # nan_mask = torch.isnan(attn_out).any(dim=1)
        #             # attn_out[nan_mask] = query_embedding[nan_mask]
                    
        #             # 4. 恢复回[1, C, H_feat, H_feat]
        #             enhanced_query_feature = attn_out.transpose(0, 1).reshape(1, self.C_feat, self.H_feat, self.W_feat)
        #             # 将enhanced_query_feature存储到对应的img1 patch位置
        #             # enhanced_feats_t_grid[feat_idx][batch, i, :, :] = enhanced_query_feature.squeeze(0)
        #             #import ipdb; ipdb.set_trace()
        #             # feats_t_grid[-1][batch, i, :, :] = enhanced_query_feature.squeeze(0)
        #             # 修改克隆后的特征图
        #             enhanced_feature[batch, i, :, :] = shortcut[batch, i, :, :] + enhanced_query_feature.squeeze(0)
                

        # '''
        # Epipolar Attention Module
        # '''
        
        # output_encoder = rearrange(enhanced_feature, "b v c h w -> (h w) (b v) c")
  
        output_decoder = self.decoder(
            output_encoder,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed,
        )  # (H X W) x B x C

        return output_decoder, output_encoder


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        img_path=None
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
                img_path=img_path
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = memory

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        # C_feat, H_feat, W_feat, num_views, pts1, v2_coord, u2_coord, index_combinations, fundamental_matrix_results,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,

    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.norm_mid = nn.LayerNorm(hidden_dim)
        self.dropout_mid = nn.Dropout(dropout)
        
        # self.C_feat = C_feat
        # self.H_feat = H_feat
        # self.W_feat = W_feat
        # self.num_views = num_views
        # self.pts1 = pts1
        # self.v2_coord = v2_coord
        # self.u2_coord = u2_coord
        # self.index_combinations = index_combinations
        # self.fundamental_matrix_results = fundamental_matrix_results
        # self.num_query = H_feat * W_feat
        # H_img, W_img = 256, 256
        # self.patch_size_h, self.patch_size_w = H_img // self.H_feat, W_img // self.W_feat

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        img_path = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src) # # (H X W) x B x C


        # B = src.shape[1] // self.num_views
        # src2 = rearrange(src, "(h w) (b v) c -> b v c h w", h=self.H_feat, v=self.num_views)

        # enhanced_feature = src2.clone()

        # '''
        # Epipolar Attention Module
        # '''
        
        # for batch in range(B):
        #     # 1. for each pair, extract the corresponding images
        #     for index_pair in self.index_combinations:
        #         # 1.1 prepare the two images
        #         i, j = index_pair
        #         img_i_path = img_path[:, i]
        #         img_j_path = img_path[:, j]
        #         img_i_name = img_i_path[batch].split('/')[-1]
        #         img_j_name = img_j_path[batch].split('/')[-1]
        #         fundamental_key = img_i_name + "_and_" + img_j_name
                
        #         if self.fundamental_matrix_results[fundamental_key] is None:
        #             continue
        #         else:
        #             # 1.1 calculate fundamental matrix
        #             Fundamental_matrix = self.fundamental_matrix_results[fundamental_key]
                    
        #             # 1.2. Epipolar line
        #             # l = F @ pt1.T, 结果 shape [3, num_query]，转置后 [num_query, 3]
        #             l = torch.matmul(Fundamental_matrix, self.pts1.t()).t()  # [num_query, 3]
        #             a = l[:, 0].unsqueeze(1)  # [num_query, 1]
        #             b = l[:, 1].unsqueeze(1)  # [num_query, 1]
        #             c = l[:, 2].unsqueeze(1)  # [num_query, 1]
                    
        #             # 1.3. Epipolar distance
        #             dist = torch.abs(a * self.u2_coord + b * self.v2_coord + c) / (a.pow(2) + b.pow(2)).sqrt()  # [num_query, H_feat*W_feat]
        #             dist = dist.reshape(self.num_query, self.H_feat, self.W_feat)  # [num_query, H_feat, W_feat]
        #             # 距离小于阈值的patch视为在极线上
        #             masks = dist < max(self.patch_size_h, self.patch_size_w) / 2  # [num_query, H_feat, W_feat]
                    
        #             # 1. Extract query embedding and image patches
        #             query_embedding = src2[batch, i, :, :].reshape(self.C_feat, self.num_query).transpose(0, 1)   # [num_query, C]
        #             img2_patches = src2[batch, j, :, :].reshape(self.C_feat, self.num_query).transpose(0, 1)  # [num_query, C]
        #             masks_flat = masks.reshape(self.num_query, -1)  # [num_query, 1024]
                    
        #             # 2. Construct Q,K,V
        #             Q = query_embedding.unsqueeze(1)  # [num_query, 1, C]
        #             K = img2_patches.unsqueeze(0).expand(self.num_query, -1, -1)  # [num_query, 1024, C]
        #             V = K  # [num_query, 1024, C]
                    
        #             # 3. 对mask为False的位置赋极小值，防止被softmax
        #             attn_mask = ~masks_flat  # [num_query, 1024], True为需要mask掉
        #             # scaled_dot_product_attention 支持 attn_mask: [num_query, 1, 1024]
        #             attn_out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask.unsqueeze(1))  # [num_query, 1, C]
        #             attn_out = attn_out.squeeze(1)  # [num_query, C]
                    
        #             # # 若某query没有极线patch（全mask），其attn_out为nan，可用原特征替换
        #             # nan_mask = torch.isnan(attn_out).any(dim=1)
        #             # attn_out[nan_mask] = query_embedding[nan_mask]
                    
        #             # 4. 恢复回[1, C, H_feat, H_feat]
        #             enhanced_query_feature = attn_out.transpose(0, 1).reshape(1, self.C_feat, self.H_feat, self.W_feat)
        #             # 将enhanced_query_feature存储到对应的img1 patch位置
        #             # enhanced_feats_t_grid[feat_idx][batch, i, :, :] = enhanced_query_feature.squeeze(0)
        #             #import ipdb; ipdb.set_trace()
        #             # feats_t_grid[-1][batch, i, :, :] = enhanced_query_feature.squeeze(0)
        #             # 修改克隆后的特征图
        #             #enhanced_feature[batch, i, :, :] = shortcut[batch, i, :, :] + enhanced_query_feature.squeeze(0)
        #             enhanced_feature[batch, i, :, :] = enhanced_query_feature.squeeze(0)
        #             #enhanced_feature[batch, i, :, :] = src2[batch, i, :, :] + self.dropout_mid(enhanced_query_feature.squeeze(0))

        # '''
        # Epipolar Attention Module
        # '''
        # enhanced_feature = rearrange(enhanced_feature, "b v c h w -> (h w) (b v) c")
        # src = src + self.dropout_mid(enhanced_feature)
        # src = self.norm_mid(enhanced_feature)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        img_path = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, img_path=img_path)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            out,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos


def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed


# ========== initializer ==========

def init_weights_normal(module, std=0.01):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            nn.init.normal_(m.weight.data, std=std)
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_xavier(module, method):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:
                nn.init.xavier_normal_(m.weight.data)
            elif "uniform" in method:
                nn.init.xavier_uniform_(m.weight.data)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def init_weights_msra(module, method):
    for m in module.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.Linear)
            or isinstance(m, nn.ConvTranspose2d)
        ):
            if "normal" in method:
                nn.init.kaiming_normal_(m.weight.data, a=1)
            elif "uniform" in method:
                nn.init.kaiming_uniform_(m.weight.data, a=1)
            else:
                raise NotImplementedError(f"{method} not supported")
            if m.bias is not None:
                m.bias.data.zero_()


def initialize(model, method, **kwargs):
    # initialize BN, Conv, & FC with different methods
    # initialize BN
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # initialize Conv & FC
    if method == "normal":
        init_weights_normal(model, **kwargs)
    elif "msra" in method:
        init_weights_msra(model, method)
    elif "xavier" in method:
        init_weights_xavier(model, method)
    else:
        raise NotImplementedError(f"{method} not supported")


def initialize_from_cfg(model, cfg):
    if cfg is None:
        initialize(model, "normal", std=0.01)
        return

    cfg = copy.deepcopy(cfg)
    method = cfg.pop("method")
    initialize(model, method, **cfg)


from argparse import Namespace
# from model.uniad_efficientnet import efficientnet_b4


class UniAD(nn.Module):
    def __init__(self, model_backbone, model_decoder, mlp_head_channels = 128):
        super(UniAD, self).__init__()
        # self.net_backbone = efficientnet_b4(pretrained=True, outblocks=[1, 5, 9, 21], outstrides=[2, 4, 8, 16], pretrained_model='model/pretrain/efficientnet-b4-6ed6700e.pth')
        self.net_backbone = get_model(model_backbone)
        self.net_merge = MFCN(inplanes=model_decoder['inplanes'], outplanes=model_decoder['outplanes'], instrides=[2, 4, 8, 16], outstrides=[16])
        self.net_ad = UniAD_decoder(inplanes=model_decoder['outplanes'], instrides=model_decoder['instrides'], feature_size=model_decoder['feature_size'],
                            # feature_jitter=Namespace(**{'scale': 20.0, 'prob': 1.0}),
                            feature_jitter=None,
                            neighbor_mask=Namespace(**{'neighbor_size': model_decoder['neighbor_size'], 'mask': [True, True, True]}),
                            # neighbor_mask=None,
                            hidden_dim=256, pos_embed_type='learned', save_recon=Namespace(**{'save_dir': 'result_recon'}),
                            initializer={'method': 'xavier_uniform'}, nhead=8, num_encoder_layers=4,
                            num_decoder_layers=4, dim_feedforward=1024, dropout=0.1, activation='relu',
                            normalize_before=False)

        # ********************* backbone {'pretrained': False, 'outblocks': [1, 5, 9, 21], 'outstrides': [2, 4, 8, 16]}
        # ********************* neck {'outstrides': [16], 'outplanes': [272], 'inplanes': [24, 32, 56, 160], 'instrides': [2, 4, 8, 16]}
        # ********************* reconstruction {'pos_embed_type': 'learned', 'hidden_dim': 256, 'nhead': 8, 'num_encoder_layers': 4, 'num_decoder_layers': 4, 'dim_feedforward': 1024, 'dropout': 0.1, 'activation': 'relu', 'normalize_before': False, 'feature_jitter': {'scale': 20.0, 'prob': 1.0}, 'neighbor_mask': {'neighbor_size': [7, 7], 'mask': [True, True, True]}, 'save_recon': {'save_dir': 'result_recon'}, 'initializer': {'method': 'xavier_uniform'}, 'feature_size': [14, 14], 'inplanes': [272], 'instrides': [16]}
        # import ipdb; ipdb.set_trace()
        self.frozen_layers = ['net_backbone']
        # self.fc_out_channels = 69632
        # self.contrastive_head = nn.Sequential(
        #     # nn.Linear(self.fc_out_channels, self.fc_out_channels),
        #     # nn.ReLU(inplace=True),
        #     nn.Linear(self.fc_out_channels, mlp_head_channels))

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats_backbone = self.net_backbone(imgs)
        feats_merge = self.net_merge(feats_backbone)
        # feats_contrast = self.net_merge(feats_backbone)
        # feats_contrast = self.contrastive_head(feats_contrast.view(feats_contrast.size(0), -1))
        # feats_contrast = F.normalize(feats_contrast, dim=1)
        # feats_merge = feats_merge.detach()
        feature_align, feature_rec, pred = self.net_ad(feats_merge)
        # feats_contrast = feature_rec
        # #import ipdb; ipdb.set_trace()
        # feats_contrast = self.contrastive_head(feats_contrast.contiguous().view(feats_contrast.size(0), -1))
        # feats_contrast = F.normalize(feats_contrast, dim=1)
        return feature_align, feature_rec, pred


@MODEL.register_module
def uniad(pretrained=False, **kwargs):
    model = UniAD(**kwargs)
    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params
    from argparse import Namespace as _Namespace

    bs = 2
    reso = 224
    x = torch.randn(bs, 3, reso, reso).cuda()

    model_backbone = _Namespace()
    model_backbone.name = 'timm_efficientnet_b4'
    model_backbone.kwargs = dict(pretrained=False, checkpoint_path='model/pretrain/efficientnet_b4_ra2_320-7eb33cd5.pth', strict=False, features_only=True, out_indices=[0, 1, 2, 3])

    net = uniad(model_backbone=model_backbone).cuda()
    net.eval()
    y = net(x)

    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))
# print(flop_count_table(FlopCountAnalysis(fn, x), max_depth=3))

