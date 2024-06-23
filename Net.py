import torch
from torch.nn import GRU, Linear
from torch_geometric.nn import MessagePassing



#现了一个类似于门控机制的效果：时间间隔越大，通过的信息越少。
class GatedMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GatedMPNN, self).__init__(aggr='add')  # "Add" aggregation #改成的最大聚合，突出聚合中心的作用
        self.lin = Linear(in_channels * 2, out_channels)
        self.gru = GRU(out_channels, in_channels)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Start propagating messages
        x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

        # Use GRU to gate the update
        x, _ = self.gru(x.unsqueeze(0), x.unsqueeze(0))
        return x.squeeze(0)

    def message(self, x_i, x_j, edge_attr):
        temp = torch.cat([x_i, x_j], dim=1)  # Shape: [E, in_channels * 2]
        out = self.lin(temp)  # Shape: [E, out_channels]
        time_diff = edge_attr  # Assuming edge_attr is the time difference
        gate = torch.sigmoid(-time_diff)  # Shape: [E, 1] #
        out = out * gate  # Element-wise multiplication, Shape: [E, out_channels]

        return out



class Net(torch.nn.Module):  #混合模型+双向LSTM
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.mpnn = GatedMPNN(num_features, 42)
        self.lstm = torch.nn.LSTM(42, 20, batch_first=True, bidirectional=True)  # 使用双向LSTM，每个方向上都有32个隐藏单元
        self.classifier = torch.nn.Linear(40, 1)  # 现在，分类器接受64个特征（双向LSTM的输出） #分类器改改？改成全链接网络？

    def forward(self, data):
        x = self.mpnn(data.x, data.edge_index, data.edge_attr)
        x, _ = self.lstm(x.unsqueeze(0))  # LSTM需要一个额外的序列维度
        x = x.squeeze(0)  # 再次移除序列维度
        return self.classifier(x)
