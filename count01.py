import pandas as pd

# 读取csv文件
df = pd.read_csv('center.csv')

# 计算标签为0和1的数量
count_normal = df[df['label'] == 0].shape[0]
count_abnormal = df[df['label'] == 1].shape[0]

# 计算异常百分比
percentage_abnormal = (count_abnormal / (count_normal + count_abnormal)) * 100

print("正常的数量: ", count_normal)
print("异常的数量: ", count_abnormal)
print("异常的百分比: ", percentage_abnormal, "%")
