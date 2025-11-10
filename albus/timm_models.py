# import timm

# models = timm.list_models()

# output = []
# for model in models:
#     if "dino" in model:
#         output.append(model)
# print(output)

# import torch
# import timm

# pth_path = "ader_weights/dino_resnet50_pretrain.pth"

# ckpt = torch.load(pth_path, map_location='cpu')

# resnet = timm.create_model('resnet50', pretrained=False, features_only=True)

# # #打印resnet参数
# # print(resnet.state_dict().keys())
# resnet.load_state_dict(ckpt, strict=True)


# import ipdb; ipdb.set_trace()
# state_dict = ckpt['model']

from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_small_patch16_dinov3.lvd1689m',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

# output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# import ipdb; ipdb.set_trace()
output = model.forward_features(transforms(img).unsqueeze(0))
# x = transforms(img).unsqueeze(0)

# x = model.patch_embed(x)
# x = model._pos_embed(x)
# # x = model.patch_drop(x)
# x = model.norm_pre(x)
# import ipdb; ipdb.set_trace()
# x= x[0]
# for blk in model.blocks:
#     x = blk(x)
# x = model.blocks(x)
# import ipdb; ipdb.set_trace()

x = model.norm(x)
