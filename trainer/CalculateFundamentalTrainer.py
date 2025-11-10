from . import TRAINER
from ._base_trainer import BaseTrainer
import numpy as np
import torch
import cv2 as cv
from MatchAnything.imcui.ui.utils import load_config, run_matching, get_matcher_zoo, get_model, filter_matches
from MatchAnything.imcui.hloc import match_dense

cfg = load_config("MatchAnything/config/config.yaml")
matcher_zoo = get_matcher_zoo(cfg['matcher_zoo'])
match_threshold = 0.1
extract_max_keypoints = 1000
keypoint_threshold = 0.015
matcher_list = 'matchanything_roma'  # 或 'matchanything_roma' 等

DEFAULT_SETTING_THRESHOLD = 0.01
DEFAULT_SETTING_MAX_FEATURES = 2000
DEFAULT_DEFAULT_KEYPOINT_THRESHOLD = 0.01
DEFAULT_ENABLE_RANSAC = True
# DEFAULT_RANSAC_METHOD = "CV2_USAC_MAGSAC"
DEFAULT_RANSAC_METHOD = "CV2_FM_RANSAC"
DEFAULT_RANSAC_REPROJ_THRESHOLD = 4
DEFAULT_RANSAC_CONFIDENCE = 0.999
DEFAULT_RANSAC_MAX_ITER = 10000
DEFAULT_MIN_NUM_MATCHES = 4
DEFAULT_MATCHING_THRESHOLD = 0.2
DEFAULT_SETTING_GEOMETRY = "Homography"
MATCHER_ZOO = None

import time

@TRAINER.register_module
class CalculateFundamentalTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(CalculateFundamentalTrainer, self).__init__(cfg)
        
    def set_input(self, inputs):
        # self.imgs = inputs['img'].cuda()
        # self.imgs_mask = inputs['img_mask'].cuda()
        # self.cls_name = inputs['cls_name']
        # self.anomaly = inputs['anomaly']
        # self.bs = self.imgs.shape[0]
        # import ipdb; ipdb.set_trace()
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        if len(self.imgs_mask.shape)==5:
            # self.imgs = self.imgs.flatten(0, 1)
            self.imgs_mask = self.imgs_mask.flatten(0, 1)
        self.cls_name = inputs['cls_name']
        self.cls_name = np.array(self.cls_name)
        self.cls_name = np.transpose(self.cls_name, (1, 0)) 
        #class_to_index = {cls: idx for idx, cls in enumerate(real_iad_classes)}
        #import ipdb; ipdb.set_trace()
        #self.contrast_labels = np.tile(np.arange(self.cls_name.shape[1]), (self.cls_name.shape[0], 1)).flatten()
        #self.contrast_labels = np.array([class_to_index[item] for item in self.cls_name.flatten()])
        
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        #import ipdb; ipdb.set_trace()
        self.img_path = np.array(self.img_path)
        self.img_path = np.transpose(self.img_path, (1, 0))
        # self.img_path = self.img_path.flatten()
  
        self.sample_anomaly = inputs['sample_anomaly']
        self.bs = self.imgs.shape[0]
        #import ipdb; ipdb.set_trace()
        num_views = 5
        class_indices = torch.arange(0, num_views) # exclude top-view
        index_combinations = torch.cartesian_prod(class_indices, class_indices) # Generate all possible pairs (including self-pairs and ordered pairs)
        mask = index_combinations[:, 0] != index_combinations[:, 1] # Create a mask to filter out pairs where i == j
        self.index_combinations = index_combinations[mask]
        
    def train(self):
        Fundamental_Matrix_Result_train = dict()
        Fundamental_Matrix_Result_test = dict()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = matcher_zoo[matcher_list]
        match_conf = model["matcher"]
        match_conf["model"]["match_threshold"] = match_threshold
        match_conf["model"]["max_keypoints"] = extract_max_keypoints
        matcher = get_model(match_conf)
        
        for data_index, train_data in enumerate(self.train_loader):
            t0 = time.time()

            self.set_input(train_data)
            for sample_path in self.img_path:
                for index_pair in self.index_combinations:
                    i, j = index_pair
                    img_i_path = sample_path[i]
                    img_j_path = sample_path[j]
                    img_i_name = img_i_path.split('/')[-1]
                    img_j_name = img_j_path.split('/')[-1]
                    
                    # Fundamental_matrix = calculate_fundamental_matrix(img_i_path, img_j_path)
                    
                    img0 = cv.imread(img_i_path)
                    img1 = cv.imread(img_j_path)
                    img0 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
                    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

                    pred = match_dense.match_images(
                        matcher, img0, img1, match_conf["preprocessing"], device=device
                    )

                    filter_matches(
                        pred,
                        ransac_method=DEFAULT_RANSAC_METHOD,
                        ransac_reproj_threshold=None,
                        ransac_confidence=None,
                        ransac_max_iter=None,
                    )

                    Fundamental_matrix = pred['geom_info']['Fundamental']
            
                    if Fundamental_matrix is not None:
                        Fundamental_matrix = torch.tensor(Fundamental_matrix, dtype=torch.float32, device=device)
                        Fundamental_Matrix_Result_train[img_i_name+"_and_"+img_j_name] = Fundamental_matrix
                    else:
                        Fundamental_Matrix_Result_train[img_i_name+"_and_"+img_j_name] = None

            t1 = time.time()
            print(f"Processed train batch {data_index+1}/{len(self.train_loader)} in {t1 - t0:.2f} seconds.")
                        
        torch.save(Fundamental_Matrix_Result_train, './matchanything_FM_RANSAC_fundamental_matrix_results_full_train.pth')
        
        for data_index, test_data in enumerate(self.test_loader):
            t0 = time.time()

            self.set_input(test_data)
            for sample_path in self.img_path:
                for index_pair in self.index_combinations:
                    i, j = index_pair
                    img_i_path = sample_path[i]
                    img_j_path = sample_path[j]
                    img_i_name = img_i_path.split('/')[-1]
                    img_j_name = img_j_path.split('/')[-1]
                    
                    #Fundamental_matrix = calculate_fundamental_matrix(img_i_path, img_j_path)
                    img0 = cv.imread(img_i_path)
                    img1 = cv.imread(img_j_path)
                    img0 = cv.cvtColor(img0, cv.COLOR_BGR2RGB)
                    img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

                    pred = match_dense.match_images(
                        matcher, img0, img1, match_conf["preprocessing"], device=device
                    )
                    filter_matches(
                        pred,
                        ransac_method=DEFAULT_RANSAC_METHOD,
                        ransac_reproj_threshold=None,
                        ransac_confidence=None,
                        ransac_max_iter=None,
                    )
                    Fundamental_matrix = pred['geom_info']['Fundamental']
            
                    if Fundamental_matrix is not None:
                        Fundamental_matrix = torch.tensor(Fundamental_matrix, dtype=torch.float32, device=device)
                        Fundamental_Matrix_Result_test[img_i_name+"_and_"+img_j_name] = Fundamental_matrix
                    else:
                        Fundamental_Matrix_Result_test[img_i_name+"_and_"+img_j_name] = None
            t1 = time.time()
            print(f"Processed test batch {data_index+1}/{len(self.test_loader)} in {t1 - t0:.2f} seconds.")
        torch.save(Fundamental_Matrix_Result_test, './matchanything_FM_RANSAC_fundamental_matrix_results_full_test.pth')
        # Save the results to a file
        torch.save(Fundamental_Matrix_Result_train + Fundamental_Matrix_Result_test, './matchanything_FM_RANSAC_fundamental_matrix_results_full.pth')