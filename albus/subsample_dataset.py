import json
import pandas as pd
import os
import copy
import random

json_path = "/home/albus/DataSets/REAL-IAD/realiad_jsons"
img_path = "/home/albus/DataSets/REAL-IAD/realiad_256"
output_path = "/home/albus/DataSets/REAL-IAD/realiad_jsons_subsample0.1"

cls_info = [product_name.split(".")[0] for product_name in os.listdir(json_path)]
cls_info.sort()

subsample_rate = 0.1

for product_name in cls_info:
    product_info = json.load(open(os.path.join(json_path, product_name + ".json"), 'r'))

    output_json = copy.deepcopy(product_info)
    
    output_json["train"] = []
    sample_list = [product_info["train"][i:i + 5] for i in range(0, len(product_info["train"]), 5)]
    random.seed(42)
    sample_index = random.sample(range(len(sample_list)), int(len(sample_list) * subsample_rate))
    for idx in sample_index:
        for view in sample_list[idx]:     
            output_json["train"].append(view)
            
    
    output_json["test"] = []
    sample_list = [product_info["test"][i:i + 5] for i in range(0, len(product_info["test"]), 5)]
    sample_anomaly_index = []
    sample_normal_index = []
    
    
    for i in range(len(sample_list)):
        has_anomaly = 0
        for view in sample_list[i]:
            if view["anomaly_class"] != "OK":
                has_anomaly = 1
        if has_anomaly == 1:
            sample_anomaly_index.append(i)
        else:
            sample_normal_index.append(i)
    
    random.seed(42)
    normal_sample_index = random.sample(sample_normal_index, int(len(sample_normal_index) * subsample_rate))
    anomaly_sample_index = random.sample(sample_anomaly_index, int(len(sample_anomaly_index) * subsample_rate))
    for index in normal_sample_index:
        for view in sample_list[index]:
            output_json["test"].append(view)
    for index in anomaly_sample_index:
        for view in sample_list[index]:
            output_json["test"].append(view)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    with open(os.path.join(output_path, product_name + ".json"), 'a') as f:
        json.dump(output_json, f)
    
    
    
    