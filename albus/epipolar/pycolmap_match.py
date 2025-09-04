import numpy as np
import pycolmap
from PIL import Image, ImageOps

output_path = 'albus/epipolar/output/'
database_path = output_path + "database.db"
image_dir = 'albus/epipolar/imgs'

pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)



# print(result)

# ransac_options = pycolmap.RANSACOptions(
#     max_error=4.0,  # for example the reprojection error in pixels
#     min_inlier_ratio=0.01,
#     confidence=0.9999,
#     min_num_trials=1000,
#     max_num_trials=100000,
# )

# answer = pycolmap.estimate_fundamental_matrix(result)       # optional dict or pycolmap.RANSACOptions)