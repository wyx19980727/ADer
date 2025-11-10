import os
import pandas as pd

input_path = 'runs/single_cls/rdepipolar_256_100e'

classes = os.listdir(input_path)

for category in classes:
    if ".csv" in category:
        classes.remove(category)

classes.sort()

print(classes)
print(len(classes))

df_list = []

for category in classes:
    class_path = os.path.join(input_path, category)
    temp_path_name = ""
    for name in os.listdir(class_path):
        if "Trainer" in name:
            temp_path_name = name
    csv_path = os.path.join(class_path, temp_path_name)
    csv_path = os.path.join(csv_path, "result.csv")
    df = pd.read_csv(csv_path, index_col=0)
    df_list.append(df)
total_df = pd.concat(df_list, axis=0)
# Add Avg
total_df.loc['Avg'] = total_df.mean()
total_df.to_csv(os.path.join(input_path, 'total_result.csv'))

    

