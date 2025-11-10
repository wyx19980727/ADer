from .dinomaly_components import vit_encoder
from .dinomaly_components.uad import ViTill
from .dinomaly_components.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2
from .dinomaly_components.vision_transformer import bMlp, Attention, LinearAttention, \
    LinearAttention2

# from .dinomaly_components.dinov3.dinov3.models.vision_transformer import SelfAttentionBlock as VitBlock

# from .dinomaly_components.dinov3.dinov3.layers import LayerScale, PatchEmbed, RMSNorm

from functools import partial

import torch.nn as nn
import torch
from torch.nn.init import trunc_normal_
from model import MODEL
import timm

from .dinomaly_components.dinov3.dinov3.hub.backbones import load_dinov3_model

REPO_DIR = "/home/albus/Python_Codes/ADer/model/dinomaly_components/dinov3"
weight_path = "/home/albus/Python_Codes/ADer/ader_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
# weight_path = "/home/albus/Python_Codes/ADer/ader_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

class Dinomaly(nn.Module):
    def __init__(self, target_layers = [2, 3, 4, 5, 6, 7, 8, 9], 
                 fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]], 
                 fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]], num_decoder_blocks=8):
        super(Dinomaly, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # encoder_name = 'dinov3_vit_small_16'
        encoder_name = 'dinov3_vits16'
        # encoder_name = 'dinov3_vitb16'
        # encoder_name = 'dinov2reg_vit_small_14'
        # encoder_name = 'dinov2reg_vit_base_14'
        # encoder_name = 'dinov2reg_vit_large_14'
        # encoder_name = 'dino_vit_small_16'

        # target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        # fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        # fuse_layer_encoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # fuse_layer_decoder = [[0, 1], [2, 3], [4, 5], [6, 7]]
        # target_layers = [8, 9, 10, 11]
        # fuse_layer_encoder = [[0, 1], [2, 3]]
        # fuse_layer_decoder = [[0, 1], [2, 3]]
        # target_layers = [5, 8, 11]
        # fuse_layer_encoder = [[0], [1], [2]]
        # fuse_layer_decoder = [[0], [4], [8]]
        # target_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # fuse_layer_encoder = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        # fuse_layer_decoder = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.num_decoder_blocks = num_decoder_blocks
        
        

        # target_layers = [11]
        # fuse_layer_encoder = [[0]]
        # fuse_layer_decoder = [[0]]

        # encoder = vit_encoder.load(encoder_name)
        # encoder = timm.create_model("vit_small_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0)
        # encoder = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=weight_path)
        encoder = load_dinov3_model(encoder_name, layers_to_extract_from=target_layers,  pretrained_weight_path=weight_path)

        # if 'small' in encoder_name:
        #     embed_dim, num_heads = 384, 6
        # elif 'base' in encoder_name:
        #     embed_dim, num_heads = 768, 12
        # elif 'large' in encoder_name:
        #     embed_dim, num_heads = 1024, 16
        #     target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        # else:
        #     raise "Architecture not in small, base, large."

        if 'vits' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'vitb' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'vitl' in encoder_name:
            embed_dim, num_heads = 1024, 16
        else:
            raise "Architecture not in vits, vitb, vitl."

        bottleneck = []
        decoder = []

        bottleneck.append(bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4))
        bottleneck = nn.ModuleList(bottleneck)

        for i in range(self.num_decoder_blocks):
            blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0.,
                        attn=LinearAttention2)
            # blk = VitBlock(dim=embed_dim, num_heads=num_heads, ffn_ratio=4.,
            #             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_drop=0.)
            #, init_values=1e-5
            # blk.attn = LinearAttention2(embed_dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.)
            decoder.append(blk)
        decoder = nn.ModuleList(decoder)


        model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=self.target_layers,
                    mask_neighbor_size=0, fuse_layer_encoder=self.fuse_layer_encoder, fuse_layer_decoder=self.fuse_layer_decoder)
        
        model = model.to(device)

        self.freeze_layer(model.encoder)

        trainable = nn.ModuleList([bottleneck, decoder])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, LayerScale):
            #     m.reset_parameters()
        # model.decoder_rope_embed._init_weights()
        self.model = model
    
    def forward(self, x, img_path=None):
        en, de, en_proj, glo_feats = self.model(x, img_path)
        return en, de, en_proj, glo_feats
    
    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

@MODEL.register_module
def dinomaly(pretrained=False, **kwargs):
    model = Dinomaly(**kwargs)
    return model