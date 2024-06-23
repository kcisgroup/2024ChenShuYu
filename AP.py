from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd
import numpy as np
from tqdm import tqdm

# 读取数据
df = pd.read_csv('UNSW_NB15_train.csv')

# 保存原始数据的全部列
original_df = df.copy()

# 准备数据
X = df.drop(['id', 'label'], axis=1).values

# 标准化输入数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将标准化后的数据再次转化为 DataFrame
df = pd.DataFrame(X, columns=df.drop(['id', 'label'], axis=1).columns)

# 将数据按顺序分层
num_layers = 50  # 层数
df['layer'] = np.repeat(np.arange(num_layers), len(df) // num_layers + 1)[:len(df)]

# 在每一层进行AP聚类，并从每个聚类中抽样
center_samples = pd.DataFrame()
silhouette_scores = []
ari_scores = []
for layer in tqdm(df['layer'].unique(), desc='层进度'):
    df_layer = df[df['layer'] == layer]
    X_layer = df_layer.drop('layer', axis=1).values

    # 应用AP聚类算法
    ap = AffinityPropagation(damping=0.9, max_iter=2000, random_state=0)
    labels_pred = ap.fit_predict(X_layer)

    # 将聚类标签添加到当前层的数据中
    df_layer['cluster_label'] = labels_pred


    # 从每个聚类中选择中心样本及其前后各两个样本
    for cluster_center_index in ap.cluster_centers_indices_:
        cluster_center_global_index = df_layer.index[cluster_center_index]

        # 确保选择的样本在DataFrame的范围内
        start_index = max(0, cluster_center_index - 2)
        end_index = min(len(df_layer) - 1, cluster_center_index + 2)

        for local_index in range(start_index, end_index + 1):
            global_index = df_layer.index[local_index]
            original_sample = original_df.loc[global_index].copy()
            original_sample['cluster_label'] = df_layer.loc[global_index, 'cluster_label']
            center_samples = pd.concat([center_samples, original_sample.to_frame().T])

#s取不同值的对比实验
    # # 从每个聚类中选择中心样本及其前后各一个样本
    # for cluster_center_index in ap.cluster_centers_indices_:
    #     cluster_center_global_index = df_layer.index[cluster_center_index]
    #
    #     # 确保选择的样本在DataFrame的范围内
    #     start_index = max(0, cluster_center_index - 1)
    #     end_index = min(len(df_layer) - 1, cluster_center_index + 1)
    #
    #     for local_index in range(start_index, end_index + 1):
    #         global_index = df_layer.index[local_index]
    #         original_sample = original_df.loc[global_index].copy()
    #         original_sample['cluster_label'] = df_layer.loc[global_index, 'cluster_label']
    #         center_samples = pd.concat([center_samples, original_sample.to_frame().T])
    #
    # # 从每个聚类中选择中心样本及其前后各三个样本
    # for cluster_center_index in ap.cluster_centers_indices_:
    #     cluster_center_global_index = df_layer.index[cluster_center_index]
    #
    #     # 确保选择的样本在DataFrame的范围内
    #     start_index = max(0, cluster_center_index - 3)
    #     end_index = min(len(df_layer) - 1, cluster_center_index + 3)
    #
    #     for local_index in range(start_index, end_index + 1):
    #         global_index = df_layer.index[local_index]
    #         original_sample = original_df.loc[global_index].copy()
    #         original_sample['cluster_label'] = df_layer.loc[global_index, 'cluster_label']
    #         center_samples = pd.concat([center_samples, original_sample.to_frame().T])

    # 计算轮廓系数
    silhouette_avg = silhouette_score(X_layer, labels_pred)
    silhouette_scores.append(silhouette_avg)

    # 计算调整兰德系数
    ari = adjusted_rand_score(original_df['label'][df['layer'] == layer].values, labels_pred)
    ari_scores.append(ari)

    print(f"第 {layer} 层的轮廓系数为: {silhouette_avg}")
    print(f"第 {layer} 层的调整兰德指数为: {ari}")

print("所有层的平均轮廓系数为 :", np.mean(silhouette_scores))
print("所有层的平均调整兰德指数为 :", np.mean(ari_scores))

center_samples.to_csv('UNSWcenter.csv', index=False)
