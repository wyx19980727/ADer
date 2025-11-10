# # import torch
# # import numpy as np
# # # a = torch.randn(114624, 1, 256, 256)
# # # print(a.element_size() * a.nelement())


# # # print(torch.cuda.is_available())
# # # t = ["bobbin"]
# # # t_array = np.array(t)
# # # print(t_array.dtype)
# # a = [torch.randn(64, 1, 256, 256), torch.randn(64, 1, 256, 256)]
# # # b = [np.array([6, 7, 8, 9, 10])]
# # print(np.concatenate(a, axis=0).shape)

# # import numpy as np

# # arr = np.array(['audiojack'],dtype='str_')

# # print(arr)

# import torch

# def check_flash_attention():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if device.type != "cuda":
#         return False

#     # 检查 GPU 架构是否支持
#     major, minor = torch.cuda.get_device_capability()
#     if major < 8:  # Ampere 架构及以上
#         return False

#     # 尝试运行 SDPA 并启用 FlashAttention
#     try:
#         q = torch.randn(1, 10, 64, device=device, dtype=torch.float16)
#         with torch.backends.cuda.sdp_kernel(enable_flash=True):
#             torch.nn.functional.scaled_dot_product_attention(q, q, q)
#         return True
#     except Exception as e:
#         print(f"FlashAttention 不可用: {e}")
#         return False

# print("FlashAttention 可用:", check_flash_attention())

import torch

path_train = './fundamental_matrix_results_train.pth'
path_test = './fundamental_matrix_results_test.pth'

fundamental_matrix_results_train = torch.load(path_train)
fundamental_matrix_results_test = torch.load(path_test)

# Merge the two dictionaries
merged_results = {**fundamental_matrix_results_train, **fundamental_matrix_results_test}
# Save the merged results to a new file
torch.save(merged_results, './fundamental_matrix_results_merged.pth')

from timm.layers.patch_embed import PatchEmbed