import torch
import torch.nn as nn
import timm
from typing import Dict, List, Tuple
from model import MODEL


class UNet(nn.Module):
    def __init__(self, encoder_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        
        # 初始化预训练编码器 (通过timm加载)
        self.encoder = timm.create_model(encoder_name, 
                                         pretrained=pretrained, 
                                         features_only=True)
        
        # 获取编码器通道数 (用于跳过连接)
        self.encoder_channels = self.encoder.feature_info.channels()
        
        # 构建对称解码器
        self.decoder = self._build_decoder()
        
        for m in self.decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 存储中间特征
        self.encoder_features = []
        self.decoder_features = []
        
        # 注册钩子捕获特征
        self._register_hooks()
        
        self.frozen_layers = ['encoder']  # 编码器通常不需要训练
    
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

    def _build_decoder(self) -> nn.ModuleDict:
        """构建与编码器对称的解码器层"""
        decoder = nn.ModuleDict()
        decoder_channels = self.encoder_channels[::-1]  # 反转通道顺序
        
        for i in range(1, len(decoder_channels)):
            # 上采样 + 卷积块
            decoder[f"up_{i}"] = nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[i-1], decoder_channels[i], 
                                   kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                # nn.Conv2d(decoder_channels[i]*2, decoder_channels[i], 
                #           kernel_size=3, padding=1),
                nn.Conv2d(decoder_channels[i], decoder_channels[i], 
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels[i]),
                nn.ReLU(inplace=True),
                nn.Conv2d(decoder_channels[i], decoder_channels[i], 
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(decoder_channels[i]),
                nn.ReLU(inplace=True)
            )
        
        # # 最终输出层
        # decoder["final"] = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        return decoder

    def _register_hooks(self):
        """为编码器和解码器注册前向钩子捕获特征图"""
        # # 编码器钩子
        # for i, block in enumerate(self.encoder):
        #     import ipdb; ipdb.set_trace()
        #     def _hook(module, input, output, idx=i):
        #         self.encoder_features[f"enc_{idx}"] = output
        #     block.register_forward_hook(_hook)
        
        # 解码器钩子
        for name, module in self.decoder.items():
            def _hook(module, input, output, layer=name):
                self.decoder_features[layer] = output
            module.register_forward_hook(_hook)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict, dict]:
        # 重置特征存储
        self.encoder_features.clear()
        self.decoder_features.clear()
        
        # 编码器前向传播 (自动触发钩子)
        enc_features = self.encoder(x)
        #import ipdb; ipdb.set_trace()
        self.encoder_features = [enc_features[i] for i in range(len(enc_features)-2)]
        
        # 解码器处理 (从最深特征开始)
        x = enc_features[-1]
        for i in range(1, len(self.encoder_channels)):
            # 上采样并拼接对应编码器特征
            x = self.decoder[f"up_{i}"][:2](x)
            # skip = enc_features[-(i+1)]
            # x = torch.cat([x, skip], dim=1)
            x = self.decoder[f"up_{i}"][2:](x)
            self.decoder_features.append(x)
        self.decoder_features = self.decoder_features[::-1][:3]
        # import ipdb; ipdb.set_trace()
            
        
        # # 最终输出
        # output = self.decoder["final"](x)
        return self.encoder_features, self.decoder_features

@MODEL.register_module
def unet(pretrained=True, **kwargs):
	model = UNet(**kwargs)
	return model


# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    model = UNet(encoder_name="resnet18", pretrained=True)
    
    input = torch.rand((8, 3, 256, 256))  # 示例输入
    encoder_features, decoder_features = model(input)
    import ipdb; ipdb.set_trace()
    
    
    
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # # 1. 初始化模型
    # model = UNetWithFeatures(encoder_name="mobilenetv3_large_100", pretrained=True)
    # model.to(device).eval()
    
    # # 2. 数据预处理 (使用timm标准变换)
    # from timm.data import create_transform
    # transform = create_transform(
    #     input_size=(3, 224, 224),
    #     mean=(0.485, 0.456, 0.406),
    #     std=(0.229, 0.224, 0.225)
    # )
    
    # # 3. 加载测试图像
    # from PIL import Image
    # img = Image.open("test_image.jpg").convert("RGB")
    # img_tensor = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]
    
    # # 4. 前向推理并获取特征
    # with torch.no_grad():
    #     output, enc_features, dec_features = model(img_tensor)
    
    # # 5. 特征可视化
    # import matplotlib.pyplot as plt
    # def visualize_feature_map(feature_map: torch.Tensor, title: str):
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(feature_map[0].mean(dim=0).cpu().numpy(), cmap='viridis')
    #     plt.title(f"{title} (Shape: {feature_map.shape})")
    #     plt.colorbar()
    #     plt.show()
    
    # # 可视化编码器第一层特征
    # visualize_feature_map(enc_features["enc_0"], "Encoder Block 0 Features")
    
    # # 可视化解码器最后一层特征
    # visualize_feature_map(dec_features["up_3"], "Decoder Block 3 Features")