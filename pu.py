from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
import hnswlib
from scipy.sparse import csr_matrix

# 加载数据
df = pd.read_csv('inter.csv')

# 数据归一化
features = ['session_duration', 'total_packets', 'total_bytes']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# 确定分层的大小
n_layers = 500
layer_size = len(df) // n_layers

# 存储所有会话及其相似度得分的列表
all_sessions_list = []

# 存储每层的代表性会话及相似性得分的列表
representative_sessions_list = []

# 定义高斯核函数转换参数
sigma = 100  # 调整sigma值为更合理的数值，可能需要根据实际情况进行调整

for layer in tqdm(range(n_layers), desc='总进度', unit='层'):
    start_idx = layer * layer_size
    end_idx = (layer + 1) * layer_size if layer < n_layers - 1 else len(df)
    df_layer = df.iloc[start_idx:end_idx].reset_index(drop=True)

    # 使用HNSW计算相似性矩阵的一部分...
    n_neighbors = 50
    dimension = len(features)
    hnsw_index = hnswlib.Index(space='l2', dim=dimension)
    hnsw_index.init_index(max_elements=len(df_layer), ef_construction=100, M=8)
    hnsw_index.add_items(df_layer[features].values)

    batch_size = 1000
    n_samples = len(df_layer)

    # 使用拉普拉斯核函数
    rows, cols, data = [], [], []
    for i in tqdm(range(0, n_samples, batch_size), desc=f'层 {layer + 1} 的相似性计算', leave=False):
        end = min(i + batch_size, n_samples)
        labels, distances = hnsw_index.knn_query(df_layer[features].values[i:end], k=n_neighbors)
        for idx, (neighbors, dists) in enumerate(zip(labels, distances)):
            for neighbor, distance in zip(neighbors, dists):
                rows.append(idx + i)
                cols.append(neighbor)
                similarity_score = np.exp(-np.abs(distance) / sigma)
                data.append(similarity_score)

    # 使用收集到的数据构建稀疏矩阵
    similarity_matrix = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples), dtype=np.float32)

    # 应用谱聚类...
    n_clusters = 200
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    cluster_labels = spectral_clustering.fit_predict(similarity_matrix)
    df_layer['cluster_label'] = cluster_labels

    # 抽取代表性会话并计算相似度得分
    for cluster_label in tqdm(range(n_clusters), desc=f'处理聚类 (层 {layer + 1})', leave=False):
        cluster_members = df_layer[df_layer['cluster_label'] == cluster_label]
        if not cluster_members.empty:
            center_degrees = similarity_matrix[cluster_members.index, :][:, cluster_members.index].sum(
                axis=1).A.flatten()
            max_center_degree_index = np.argmax(center_degrees)
            representative_session = cluster_members.iloc[max_center_degree_index].copy()
            representative_session['similarity_score'] = center_degrees[max_center_degree_index]
            representative_sessions_list.append(representative_session)

            # 更新所有会话列表，添加相似度得分
            for i, local_index in enumerate(cluster_members.index):
                df_layer.loc[local_index, 'similarity_score'] = center_degrees[i]

    all_sessions_list.append(df_layer)

# 将所有层的数据合并
all_sessions_with_scores_df = pd.concat(all_sessions_list, ignore_index=True)

# 显示或保存
all_sessions_with_scores_df.to_csv('all_sessions_with_similarity_scores.csv', index=False)


