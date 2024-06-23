import pandas as pd
import itertools

df_test = pd.read_csv('/home/csy/PycharmProjects/A/UNSW_NB15/UNSW_NB15_test.csv')

# 初始化源节点和目标节点列表
source_nodes_test = []
target_nodes_test = []
# 初始化中心点列表
center_points_test = []

# 在测试集中找到每个单位中的中心点，再找到这些中心点前后各两个点，形成5个点的单位，单位内的点全部相连
for center_index in range(2, len(df_test), 5):  # 假定第3、8...号为中心
    unit_indices = list(range(center_index - 2, min(len(df_test), center_index + 3)))
    if len(unit_indices) == 5:  # 确保只有当单位包含5个点时才添加边
        unit_edges = list(itertools.combinations(unit_indices, 2))
        for edge in unit_edges:
            source_nodes_test.append(df_test.iloc[edge[0]])  # 使用df_test中的完整行作为节点
            target_nodes_test.append(df_test.iloc[edge[1]])  # 使用df_test中的完整行作为节点
        center_points_test.append(df_test.iloc[center_index])  # 将中心点加入列表

# 将中心点列表转化为DataFrame
df_center_points_test = pd.DataFrame(center_points_test)

# 将DataFrame保存为CSV文件
df_center_points_test.to_csv('testcenter.csv', index=False)
