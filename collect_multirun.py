import os
import pandas as pd

# --- 配置 ---
input_path = 'runs/'
prefix = "Dinomaly_small_loosecosloss_size256_dinov3_officialloadhub_freeze_decodernorope_drop0_fusefinallayers"
output_csv_path = "all_dinolayer_experiments_summary.csv" # 新建的汇总文件名

# --- 主逻辑 ---

# 1. 查找所有符合条件的实验文件夹
all_items = os.listdir(input_path)
dinolayers_list = []
for item_name in all_items:
    # 确保是文件夹并且包含特定前缀
    if os.path.isdir(os.path.join(input_path, item_name)) and prefix in item_name:
        dinolayers_list.append(item_name)

print(f"找到了 {len(dinolayers_list)} 个匹配的实验文件夹。")
print(dinolayers_list)

# 用于存储每一行结果（作为Series）的列表
results_list = []

# 2. 遍历每个实验文件夹，提取结果
for dinolayer_exp in dinolayers_list:
    exp_path = os.path.join(input_path, dinolayer_exp)
    csv_path = os.path.join(exp_path, "result.csv")

    try:
        # 读取CSV文件，并将第一列作为索引
        df = pd.read_csv(csv_path, index_col=0)
        
        # 提取最后一行数据，这将是一个Pandas Series
        last_row_series = df.iloc[-1]
        
        # 关键：将这个Series的 .name 属性设置为实验的名称
        # 这在后续合并成DataFrame时，会自动成为新DataFrame的行索引
        last_row_series.name = dinolayer_exp
        
        # 将处理后的Series添加到列表中
        results_list.append(last_row_series)
        
        print(f"成功处理: {dinolayer_exp}")

    except FileNotFoundError:
        print(f"警告: 在文件夹 '{dinolayer_exp}' 中未找到 result.csv，已跳过。")
    except Exception as e:
        print(f"处理 '{dinolayer_exp}' 时发生错误: {e}")

# 3. 将所有结果合并成一个DataFrame并保存
if results_list:
    # 从Series列表中创建DataFrame
    # 每个Series会成为DataFrame的一行
    summary_df = pd.DataFrame(results_list)
    
    # 给索引列（即我们的实验名）设置一个名字，这将成为CSV文件中的表头
    summary_df.index.name = "dinolayer_exp"
    
    # 将汇总的DataFrame保存到新的CSV文件中
    summary_df.to_csv(output_csv_path)
    
    print("\n--- 操作完成 ---")
    print("汇总的DataFrame:")
    print(summary_df)
    print(f"\n结果已成功保存到: {output_csv_path}")
else:
    print("\n没有找到任何可处理的数据。")