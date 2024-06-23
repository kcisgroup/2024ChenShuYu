import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取CSV数据
data = pd.read_csv('all_sessions_with_similarity_scores.csv')

# 数据归一化
scaler = MinMaxScaler()
data[['total_bytes', 'total_packets', 'similarity_score']] = scaler.fit_transform(data[['total_bytes', 'total_packets', 'similarity_score']])
# 创建一个图
G = nx.Graph()

# 遍历数据的每一行，添加节点和边
for index, row in data.iterrows():
    # 用户节点和应用节点的标签
    user_node = 'user_' + str(row['src_ipv4'])
    app_node = 'app_' + str(row['dst_ipv4']) + ':' + str(row['dst_port'])

    # 使用相似度得分调整权重
    weight = (row['total_bytes'] + row['total_packets']) * row['similarity_score']
    # weight = (row['total_bytes'] + row['total_packets'])

    # 如果边已经存在，增加权重；否则，添加新边
    if G.has_edge(user_node, app_node):
        G[user_node][app_node]['weight'] += weight
    else:
        G.add_edge(user_node, app_node, weight=weight)

# 使用Louvain方法找到最佳分区
import community as community_louvain
partition = community_louvain.best_partition(G)

# 过滤掉应用节点的社区分类，仅保留用户节点的分类
user_partition = {node: part for node, part in partition.items() if 'user_' in node}

community_dict = {}


for node, community_id in user_partition.items():
    if community_id not in community_dict:
        community_dict[community_id] = []
    community_dict[community_id].append(node)

## 计算每个社区中user_node的数量
community_user_sizes = {community: len([node for node in nodes if 'user_' in node]) for community, nodes in community_dict.items()}

# 阈值设置为2，对于那些user_node数量小于2的社群
threshold = 2

# 为每个小社区找到一个合适的较大社区进行合并
for small_community, size in community_user_sizes.items():
    if size < threshold:
        min_distance = float('inf')
        closest_community = None

        for larger_community, nodes in community_dict.items():
            # 确保目标社群至少包含threshold数量的user_node
            if len([node for node in nodes if 'user_' in node]) >= threshold:
                distance = abs(small_community - larger_community)
                if distance < min_distance:
                    min_distance = distance
                    closest_community = larger_community

        # 合并小社区到最接近的较大社区
        community_dict[closest_community].extend(community_dict[small_community])
        del community_dict[small_community]

# 输出每个社区的节点
for community_id, nodes in community_dict.items():
    print(f"Community {community_id}:")
    for node in nodes:
        # 移除 'user_' 前缀并输出
        cleaned_node = node.replace('user_', '')
        print(f"{cleaned_node}")
    print("\n")

# 输出节点和边的数量
print(f"Number of user nodes: {len([node for node in G.nodes() if 'user_' in node])}")
print(f"Number of app nodes: {len([node for node in G.nodes() if 'app_' in node])}")
print(f"Number of edges: {len(G.edges())}")

