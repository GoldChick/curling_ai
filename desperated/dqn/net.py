import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #初始化网络
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(80, 256)         # 定义全连接层1
        self.fc1.weight.data.normal_(0, 0.1)  # 按(0, 0.1)的正态分布初始化权重
        self.fc2 = nn.Linear(256, 1024)       # 定义全连接层2
        self.fc2.weight.data.normal_(0, 0.1)  # 按(0, 0.1)的正态分布初始化权重
        self.out = nn.Linear(1024, 1600)      # 定义输出层
        self.out.weight.data.normal_(0, 0.1)  # 按(0, 0.1)的正态分布初始化权重
        
    #网络前向推理
    def forward(self, x):
        x = self.fc1(x)                       # 输入张量经全连接层1传递
        x = F.relu(x)                         # 经relu函数激活
        x = self.fc2(x)                       # 经全连接层2传递
        x = F.relu(x)                         # 经relu函数激活
        return self.out(x)                    # 经输出层传递得到输出张量