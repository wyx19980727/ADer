import torch

train_path = "./matchanything_FM_RANSAC_fundamental_matrix_results_full_train.pth"
test_path = "./matchanything_FM_RANSAC_fundamental_matrix_results_full_test.pth"

train_load = torch.load(train_path)
test_load = torch.load(test_path)

merged_dict = {**train_load, **test_load}
torch.save(merged_dict, "./matchanything_FM_RANSAC_fundamental_matrix_results_full.pth")