from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'convnext_tiny.dinov3_lvd1689m',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

import ipdb; ipdb.set_trace()

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 96, 56, 56])
    #  torch.Size([1, 192, 28, 28])
    #  torch.Size([1, 384, 14, 14])
    #  torch.Size([1, 768, 7, 7])

    print(o.shape)