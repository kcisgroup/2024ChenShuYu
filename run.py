import itertools
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch.optim import lr_scheduler
from model.Net import Net #TGGNN-BiLSTM
from model.GGNN import GGNN
from model.Netnobian import Netnobian
from model.NetLSTM import NetLSTM
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def train_and_test(epochnum):
    # 读取APcenter5.csv数据
    df = pd.read_csv('APcenter5.csv')

   # 读取center.csv数据
    center_df = pd.read_csv('UNSWcenter.csv')

    # 获取在APcenter5.csv中出现的center.csv的数据的id
    center_ids = center_df['id'].tolist()
    df_center_ids = df[df['id'].isin(center_ids)]['id'].tolist()

    # 获取center.csv数据在APcenter5.csv中的顺序
    ordered_center_ids = [id for id in center_ids if id in df_center_ids]

    # 获取df中所有id的列表
    df_id_list = df['id'].tolist()

    # 获取ordered_center_ids在df中的索引
    ordered_center_indices = [df_id_list.index(id) for id in ordered_center_ids if id in df_id_list]

    # 初始化节点列表
    source_nodes = []
    target_nodes = []

    # 按照center.csv数据的顺序连接前后两个节点
    for i in range(len(ordered_center_indices) - 1):
        source_nodes.append(ordered_center_indices[i])
        target_nodes.append(ordered_center_indices[i + 1])

    # 在APcenter5.csv中找到center.csv的数据点，再找到这些数据点前后各两个点，形成5个点的单位，单位内的点全部相连
    for center_index in ordered_center_indices:  # 修改为索引
        unit_indices = list(range(max(0, center_index - 2), min(len(df_id_list), center_index + 3)))
        if len(unit_indices) == 5:  # 确保只有当单位包含5个点时才添加边
            unit_edges = list(itertools.permutations(unit_indices, 2))
            for edge in unit_edges:
                source_nodes.append(edge[0])
                target_nodes.append(edge[1])

    # 标准化边属性，使用节点间的id差值的绝对值
    edge_attr = np.abs(np.array(source_nodes) - np.array(target_nodes))
    edge_attr_mean = edge_attr.mean()
    edge_attr_std = edge_attr.std()
    edge_attr_normalized = (edge_attr - edge_attr_mean) / edge_attr_std

    # 转换为torch.Tensor
    source_nodes = torch.tensor(source_nodes, dtype=torch.long)
    target_nodes = torch.tensor(target_nodes, dtype=torch.long)
    edge_index = torch.stack((source_nodes, target_nodes), dim=0)
    edge_attr = torch.tensor(edge_attr_normalized, dtype=torch.float).view(-1, 1)

    # 其他数据处理部分和原来的代码保持不变
    labels = df['label'].values
    df = df.drop(columns=['id', 'label'])
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    x = torch.tensor(df, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.float)

    # 创建图数据
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    df_test = pd.read_csv('UNSW_NB15_test.csv')

    # 初始化源节点和目标节点列表
    source_nodes_test = []
    target_nodes_test = []


    # 在测试集中找到每个单位中的中心点，再找到这些中心点前后各两个点，形成5个点的单位，单位内的点全部相连
    for center_index in range(2, len(df_test), 5):  # 假定第3、8...号为中心
        unit_indices = list(range(center_index - 2, min(len(df_test), center_index + 3)))
        if len(unit_indices) == 5:  # 确保只有当单位包含5个点时才添加边
            # unit_edges = list(itertools.combinations(unit_indices, 2))
            unit_edges = list(itertools.permutations(unit_indices, 2))
            for edge in unit_edges:
                source_nodes_test.append(edge[0])
                target_nodes_test.append(edge[1])


    # 按照顺序连接中心点
    for i in range(2, len(df_test) - 5, 5):
        source_nodes_test.append(i)
        target_nodes_test.append(i + 5)

    # 标准化边属性，使用节点间的id差值的绝对值
    edge_attr_test = np.abs(np.array(source_nodes_test) - np.array(target_nodes_test))
    edge_attr_mean_test = edge_attr_test.mean()
    edge_attr_std_test = edge_attr_test.std()
    edge_attr_normalized_test = (edge_attr_test - edge_attr_mean_test) / edge_attr_std_test

    # 转换为torch.Tensor
    source_nodes_test = torch.tensor(source_nodes_test, dtype=torch.long)
    target_nodes_test = torch.tensor(target_nodes_test, dtype=torch.long)
    edge_index_test = torch.stack((source_nodes_test, target_nodes_test), dim=0)
    edge_attr_test = torch.tensor(edge_attr_normalized_test, dtype=torch.float).view(-1, 1)

    # 其他数据处理部分和原来的代码保持不变
    labels_test = df_test['label'].values
    df_test = df_test.drop(columns=['id', 'label'])
    scaler_test = StandardScaler()
    df_test = scaler_test.fit_transform(df_test)
    x_test = torch.tensor(df_test, dtype=torch.float)
    y_test = torch.tensor(labels_test, dtype=torch.float)

    # 创建图数据
    test_data = Data(x=x_test, edge_index=edge_index_test, edge_attr=edge_attr_test, y=y_test)

    #模型，优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(data.num_features).to(device) #更改模型处
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    # 创建一个学习率调度器，每10个epoch衰减一次学习率
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()
    for epoch in range(epochnum):
        optimizer.zero_grad()
        out = model(data).squeeze(1)
        pos_weight = torch.tensor([5/5])  # 这是异常类和非异常类的比例
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(out, data.y) #改为更注意正类的样本

        loss.backward()
        optimizer.step()
        scheduler.step()

        # 添加对训练集学习程度的指标
        model.eval()
        with torch.no_grad():
            out_train = model(data).squeeze(1)
        probs_train = torch.sigmoid(out_train)
        auc_score_train = roc_auc_score(data.y.cpu().numpy(), probs_train.cpu().numpy())
        predictions_train = (probs_train > 0.8).cpu().numpy()  # 使用默认阈值0.5
        acc_train = accuracy_score(data.y.cpu().numpy(), predictions_train)
        f1_train = f1_score(data.y.cpu().numpy(), predictions_train)
        precision_train = precision_score(data.y.cpu().numpy(), predictions_train)
        recall_train = recall_score(data.y.cpu().numpy(), predictions_train)

        print( f'Epoch: {epoch + 1}, Loss: {loss.item()}, AUC: {auc_score_train}, Accuracy: {acc_train}, F1: {f1_train}, Precision: {precision_train}, Recall: {recall_train}')
        model.train()

    # 预测测试数据
    model.eval()
    with torch.no_grad():
        out_test = model(test_data).squeeze(1)

    # 使用sigmoid函数将输出转换为概率
    probs_test = torch.sigmoid(out_test)

    # 计算Precision和Recall的不同阈值下的值
    precision, recall, thresholds = precision_recall_curve(test_data.y.cpu().numpy(), probs_test.cpu().numpy())

    # 绘制Precision-Recall曲线
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()

    threshold = 0.8
    auc_score = roc_auc_score(test_data.y.cpu().numpy(), probs_test.cpu().numpy())
    predictions = (probs_test > threshold).cpu().numpy()
    acc = accuracy_score(test_data.y.cpu().numpy(), predictions)
    f1 = f1_score(test_data.y.cpu().numpy(), predictions)
    precision = precision_score(test_data.y.cpu().numpy(), predictions)
    recall = recall_score(test_data.y.cpu().numpy(), predictions)

    return auc_score, acc, f1, precision, recall


auc_scores = []
acc_scores = []
f1_scores = []
precision_scores = []
Recall_scores = []

# 运行n次
for i in range(30):
    # 运行算法
    auc, acc, f1, precision, recall= train_and_test(10)
    # 打印每次运行的结果
    print('Run: {}, AUC: {:.4f}, ACC: {:.4f}, F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(i+1,auc, acc, f1, precision, recall))
    # 记录auc和f1分数
    auc_scores.append(auc)
    acc_scores.append(acc)
    f1_scores.append(f1)
    precision_scores.append(precision)
    Recall_scores.append(recall)

# 计算平均auc和f1分数
average_auc = np.mean(auc_scores)
average_acc = np.mean(acc_scores)
average_f1 = np.mean(f1_scores)
average_precision = np.mean(precision_scores)
average_recall = np.mean(Recall_scores)

# 计算auc和f1分数的方差
variance_acc = np.var(acc_scores)
variance_f1 = np.var(f1_scores)

# 打印平均auc和f1分数以及方差
print('Average AUC: {:.4f}, Average ACC: {:.4f}, Variance ACC: {:.4f}, Average F1: {:.4f}, Variance F1: {:.4f}, Average Precision: {:.4f}, Average Recall: {:.4f}'.format(average_auc, average_acc, variance_acc, average_f1, variance_f1, average_precision, average_recall))




