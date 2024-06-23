import pandas as pd

# 读取CSV文件
df = pd.read_csv('UNSWcenter7.csv')

# 重复数据是基于'id'列，删除重复的行
df.drop_duplicates(subset='id', keep='first', inplace=True)

# 删除名为'cluster_label'的列
if 'cluster_label' in df.columns:
    df = df.drop('cluster_label', axis=1)

# 保存处理过的数据到新的CSV文件
df.to_csv('UNSWcenter77.csv', index=False)
