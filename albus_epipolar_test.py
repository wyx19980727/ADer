from util.data import get_img_loader
from data.utils import get_transforms

import torchvision.transforms.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

# # 反归一化 img1 以便可视化
# def denormalize(tensor, mean, std):
#     mean = torch.tensor(mean).reshape(-1, 1, 1)
#     std = torch.tensor(std).reshape(-1, 1, 1)
#     return tensor * std + mean

# img1_vis = denormalize(img1[0].cpu(), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
# img1_vis = img1_vis.permute(1, 2, 0).numpy()  # [H, W, 3]
# img1_vis = (img1_vis * 255).clip(0, 255).astype('uint8')

# # 可视化patch边界和中心点
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(img1_vis)

# # 画patch边界
# for i in range(H_feat):
#     for j in range(W_feat):
#         y0 = int(i * patch_size_h)
#         x0 = int(j * patch_size_w)
#         rect = plt.Rectangle((x0, y0), patch_size_w, patch_size_h, edgecolor='lime', facecolor='none', linewidth=0.5)
#         ax.add_patch(rect)

# # 画中心点
# centers_x = query_points[0, 0].flatten()
# centers_y = query_points[0, 1].flatten()
# ax.scatter(centers_x, centers_y, c='red', s=8, marker='x', label='patch center')

# ax.set_title('Teacher patches and their centers')
# plt.legend()
# plt.tight_layout()
# fig.savefig("teacher_patches_centers.png")

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
	F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)
	return F

def draw_epiline(img1_path, img2_path, F):
    """
    在img2上绘制img1中心点对应的极线。
    """
    # 以彩色模式加载图片用于绘制
    img1 = cv.imread(img1_path, cv.IMREAD_COLOR)
    img2 = cv.imread(img2_path, cv.IMREAD_COLOR)

    # img1 = cv.resize(img1, (640,480))
    # img2 = cv.resize(img2, (640,480))
    
    # 获取图像尺寸
    h, w = img1.shape[:2]
    
    # 1. 定义img1的中心点
    # 注意点的格式需要是1x1x2的Numpy数组，类型为float32
    center_point = np.array([[[w / 2, h / 2]]], dtype=np.float32)

    # 在img1上绘制中心点
    cv.circle(img1, (int(w/2), int(h/2)), 10, (0, 255, 0), -1)

    # 2. 计算img1中心点在img2中对应的极线
    # lines2是一个N x 1 x 3的数组，每行是[a, b, c]，代表极线方程 ax + by + c = 0
    lines2 = cv.computeCorrespondEpilines(center_point, 1, F)
    line = lines2[0][0]

    # 3. 绘制极线
    # 为了绘制直线，我们需要找到直线上的两个点
    # 我们可以通过令 x=0 和 x=w-1 来计算对应的y值
    x0, y0 = 0, int(-line[2] / line[1])
    x1, y1 = w - 1, int(-(line[2] + line[0] * (w - 1)) / line[1])
    
    # 在img2上绘制极线
    cv.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 2) # 绿色，宽度为2

    # 4. 显示结果
    # 将BGR图像转换为RGB以便matplotlib正确显示
    img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
    img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(img1_rgb)
    axes[0].set_title('Image 1 with Center Point')
    axes[0].axis('off')
    
    axes[1].imshow(img2_rgb)
    axes[1].set_title('Image 2 with Epipolar Line')
    axes[1].axis('off')
    
    plt.savefig('epipolar_lines.png')
    
def get_patches_on_epiline(teacher_features, F, img1_center_pt, input_size):
    """
    获取img2特征图上，位于img1中心点对应极线上的所有patches。

    Args:
        teacher_features (list of torch.Tensor): 模型从img2生成的特征图列表。
        F (np.ndarray): 基本矩阵。
        img1_center_pt (np.ndarray): img1的中心点坐标。
        input_size (int): 输入到模型的图像尺寸。

    Returns:
        dict: 一个字典，键是特征图的层级索引，值是包含'coords'和'patches'的字典。
    """
    print("正在提取位于极线上的特征 patches...")
    
    # 1. 计算img1中心点在img2中对应的极线方程 (ax + by + c = 0)
    epiline = cv.computeCorrespondEpilines(img1_center_pt, 1, F)[0][0]
    a, b, c = epiline
    
    # 用于存储结果的字典
    all_patches_on_line = {}

    # 2. 遍历模型输出的每一个层级的特征图
    for stage_idx, feature_map in enumerate(teacher_features):
        B, C, H_feat, W_feat = feature_map.shape
        
        # 3. 计算从特征图坐标到原始图像坐标的步长(stride)
        stride = input_size / H_feat
        
        patches_info = {'coords': [], 'patches': []}
        
        # 4. 遍历特征图上的每一个位置(patch)
        for fy in range(H_feat):
            for fx in range(W_feat):
                # 5. 将特征图坐标(fx, fy)映射回原始图像的中心坐标(ix, iy)
                ix = (fx + 0.5) * stride
                iy = (fy + 0.5) * stride
                
                # 6. 计算该点到极线的归一化距离
                # dist = |a*ix + b*iy + c| / sqrt(a^2 + b^2)
                dist = abs(a * ix + b * iy + c) / np.sqrt(a**2 + b**2)
                
                # 7. 判断距离是否小于阈值（这里用步长的一半作为阈值）
                # 这意味着极线必须穿过该patch在原图上对应的区域
                if dist < (stride / 2):
                    # 如果满足条件，则保存该patch的坐标和特征向量
                    patch_coords = (fy, fx)
                    patch_vector = feature_map[0, :, fy, fx].detach().cpu().numpy()
                    
                    patches_info['coords'].append(patch_coords)
                    patches_info['patches'].append(patch_vector)

        all_patches_on_line[stage_idx] = patches_info
        print(f"  - 在特征层 {stage_idx} (尺寸: {H_feat}x{W_feat}, 步长: {stride}): "
              f"找到 {len(patches_info['coords'])} 个 patches。")
        
    return all_patches_on_line

def visualize_patches_on_feature_maps(img2_path, F, img1_center_pt, patches_data, input_size):
    """可视化原始图像上的极线和特征图上被选中的patches。"""
    #import ipdb; ipdb.set_trace()
    
    # --- Part 1: 在原始图像上绘制极线 ---
    img2_orig = cv.imread(img2_path)
    h, w, _ = img2_orig.shape
    
    line = cv.computeCorrespondEpilines(img1_center_pt, 1, F)[0][0]
    a, b, c = line
    x0, y0 = 0, int(-c / b)
    x1, y1 = w - 1, int(-(c + a * (w - 1)) / b)
    
    img2_with_line = cv.line(img2_orig, (x0, y0), (x1, y1), (0, 0, 255), 2)
    img2_with_line = cv.cvtColor(img2_with_line, cv.COLOR_BGR2RGB)

    # --- Part 2: 在特征图上高亮显示patches ---
    num_stages = len(patches_data)
    fig, axes = plt.subplots(1, num_stages + 1, figsize=(5 * (num_stages + 1), 5))

    axes[0].imshow(img2_with_line)
    axes[0].set_title('Image 2 with Epipolar Line')
    axes[0].axis('off')

    for stage_idx, info in patches_data.items():
        stride = 2**(stage_idx + 2) # ResNet strides: 4, 8, 16
        feat_size = int(input_size // stride)
        
        # 创建一个空白的特征图可视化
        vis_map = np.zeros((feat_size, feat_size), dtype=np.float32)
        
        # 高亮选中的patches
        for y, x in info['coords']:
            # import ipdb; ipdb.set_trace()
            vis_map[y, x] = 1.0
        
        axes[stage_idx + 1].imshow(vis_map, cmap='viridis')
        axes[stage_idx + 1].set_title(f'Stage {stage_idx} Highlighted Patches')
        axes[stage_idx + 1].axis('off')
        
    plt.tight_layout()
    plt.savefig("epipolar_patches_visualization.png")
    print("\n可视化结果已保存到 'epipolar_patches_visualization.png'")
    plt.show()

# 0. preapare the two images

size = 256
# img1_path = '/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C2_20231021130235.jpg'
# img2_path = '/home/albus/DataSets/REAL-IAD/realiad_256/audiojack/OK/S0001/audiojack_0001_OK_C3_20231021130235.jpg'

img1_path = '/home/albus/DataSets/REAL-IAD/realiad_256/sim_card_set/OK/S0001/sim_card_set_0001_OK_C3_20230922140928.jpg'
img2_path = '/home/albus/DataSets/REAL-IAD/realiad_256/sim_card_set/OK/S0001/sim_card_set_0001_OK_C4_20230922140928.jpg'

loader = get_img_loader('pil')

img1 = loader(img1_path)
img2 = loader(img2_path)

train_transforms = [
	dict(type='ToTensor'),
	dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
]

transform = get_transforms('', train=True, cfg_transforms=train_transforms)

img1 = transform(img1)
img1 = img1.unsqueeze(0)  # Add batch dimension
img2 = transform(img2)
img2 = img2.unsqueeze(0)  # Add batch dimension

model = timm.create_model('resnet18', 
                  pretrained=True, 
                  features_only=True,
                  out_indices=(1, 2, 3, 4))

teacher_features = model(img2)

# 1. Generate the fundamental matrix by opencv sift and 8-point algorithm
F = calculate_fundamental_matrix(img1_path, img2_path)
#F = np.array([[-2.9470707929872554e-05, 4.497253462022785e-05, 0.009321176180277824], [-7.344026805882998e-05, 2.26290259531242e-05, 0.02375064904293248], [0.001767815030926779, -0.023707022780805064, -0.4364002920233576]])
#F = np.array([[-2.4554061707712235e-05, 0.00016435427363360952, -0.003734527450624095], [-0.0002196998458254249, 9.443372433169353e-06, 0.04782378964190472], [0.013884213702039608, -0.03218369942574307, -1.2660561070995056]])
# 2. Draw epipolar lines
draw_epiline(img1_path, img2_path, F)

# 3. 获取极线上的所有patches
B, C, H, W = img1.shape
img1_center = np.array([[[W / 2, H / 2]]], dtype=np.float32)

patches_on_line = get_patches_on_epiline(teacher_features, F, img1_center, size)
# 3. (可选) 可视化结果
visualize_patches_on_feature_maps(img2_path, F, img1_center, patches_on_line, size)




# import ipdb; ipdb.set_trace()