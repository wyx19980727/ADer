import torch

path = './fundamental_matrix_results_full.pth'
fundamental_matrix_results = torch.load(path)

for key, value in fundamental_matrix_results.items():
    if value is not None:
        if not value.shape  == (3, 3):
            print(f"Key: {key} has shape {value.shape}, expected (3, 3)")
    else:
        print(f"Key: {key} has None value")