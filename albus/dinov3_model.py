# import torch

# REPO_DIR = "/home/albus/Python_Codes/ADer/model/dinomaly_components/dinov3"

# weight_path = "/home/albus/Python_Codes/ADer/ader_weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"

# dinov3_convnext_tiny = torch.hub.load(REPO_DIR, 'dinov3_convnext_tiny', source='local', weights=weight_path)

# print(dinov3_convnext_tiny)

# x = torch.randn(1, 3, 256, 256)
# outputs = []

# for i in range(4):
#     x = dinov3_convnext_tiny.downsample_layers[i](x)
#     x = dinov3_convnext_tiny.stages[i](x)
#     outputs.append(x)

# for output in outputs:
#     print(output.shape)

import torch

REPO_DIR = "/home/albus/Python_Codes/ADer/model/dinomaly_components/dinov3"
weight_path = "/home/albus/Python_Codes/ADer/ader_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# 1. 加载不带预训练权重的模型结构
#    通过将 'weights' 参数设置为 None 或不提供该参数来实现
encoder_model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local', weights=weight_path)
model_state_dict = encoder_model.state_dict()

# 2. 加载预训练权重
pretrained_weights = torch.load(weight_path, map_location='cpu')

# state_dict 可能嵌套在字典的某个键下，需要根据实际情况调整
# 常见的键包括 'model', 'state_dict' 等
if 'model' in pretrained_weights:
    pretrained_state_dict = pretrained_weights['model']
elif 'state_dict' in pretrained_weights:
    pretrained_state_dict = pretrained_weights['state_dict']
else:
    pretrained_state_dict = pretrained_weights

# 3. 比较 state_dict 的键
model_keys = set(model_state_dict.keys())
pretrained_keys = set(pretrained_state_dict.keys())

# 找出在模型中但不在预训练权重中的键 (缺失的键)
missing_keys = model_keys - pretrained_keys
if missing_keys:
    print("缺失的键 (Missing keys):")
    for key in sorted(list(missing_keys)):
        print(f"  {key}")

# 找出在预训练权重中但不在模型中的键 (意外的键)
unexpected_keys = pretrained_keys - model_keys
if unexpected_keys:
    print("\n意外的键 (Unexpected keys):")
    for key in sorted(list(unexpected_keys)):
        print(f"  {key}")

# 检查是否有形状不匹配的参数
mismatched_keys = []
for key in model_keys.intersection(pretrained_keys):
    if model_state_dict[key].shape != pretrained_state_dict[key].shape:
        mismatched_keys.append((key, model_state_dict[key].shape, pretrained_state_dict[key].shape))

if mismatched_keys:
    print("\n形状不匹配的键 (Mismatched shape keys):")
    for key, model_shape, pretrained_shape in mismatched_keys:
        print(f"  {key}: 模型中的形状 {model_shape}, 预训练权重中的形状 {pretrained_shape}")

if not missing_keys and not unexpected_keys and not mismatched_keys:
    print("\n所有键完全匹配！")
    # 如果所有键都匹配，可以安全地加载权重
    encoder_model.load_state_dict(pretrained_state_dict, strict=True)
else:
    # 如果存在不匹配，使用 strict=False 加载匹配的权重，并会收到警告
    print("\n由于存在不匹配的键，建议使用 strict=False 进行加载。")
    encoder_model.load_state_dict(pretrained_state_dict, strict=False)