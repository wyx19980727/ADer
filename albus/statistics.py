import json
import pandas as pd
import os

json_path = "/home/albus/DataSets/REAL-IAD/realiad_jsons"
img_path = "/home/albus/DataSets/REAL-IAD/realiad_256"
cls_info = [product_name.split(".")[0] for product_name in os.listdir(json_path)]
cls_info.sort()
df = pd.DataFrame(columns=["product_name", "num_train_normal", "num_train_abnormal", "num_test_normal", "num_test_abnormal", "num_test_total"])
for product_name in cls_info:
    product_info = json.load(open(os.path.join(json_path, product_name + ".json"), 'r'))
    normal_class = product_info["meta"]["normal_class"]
    num_train_normal = 0
    num_train_abnormal = 0
    num_test_normal = 0
    num_test_abnormal = 0
    for img in product_info["train"]:
        if img["anomaly_class"] == normal_class:
            num_train_normal += 1
        else:
            num_train_abnormal += 1
    for img in product_info["test"]:
        if img["anomaly_class"] == normal_class:
            num_test_normal += 1
        else:
            num_test_abnormal += 1
    num_test_total = num_test_normal + num_test_abnormal
    df = pd.concat([df, pd.DataFrame({"product_name": [product_name], "num_train_normal": [num_train_normal], "num_train_abnormal": [num_train_abnormal], "num_test_normal": [num_test_normal], "num_test_abnormal": [num_test_abnormal], "num_test_total": [num_test_total]})], ignore_index=True)
df = pd.concat([df, pd.DataFrame({"product_name": ["total"], "num_train_normal": [df["num_train_normal"].sum()], "num_train_abnormal": [df["num_train_abnormal"].sum()], "num_test_normal": [df["num_test_normal"].sum()], "num_test_abnormal": [df["num_test_abnormal"].sum()], "num_test_total": [df["num_test_total"].sum()]})], ignore_index=True)
df.to_csv("realiad_info.csv", index=False)
print(df)
