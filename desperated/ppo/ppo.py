import torch.nn as nn
import torch
import numpy as np

# 创建Actor网络类继承自nn.Module


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(80, 128)       # 定义全连接层1
        self.fc2 = nn.Linear(128, 64)       # 定义全连接层2
        self.fc3 = nn.Linear(64, 10)        # 定义全连接层3
        self.out = nn.Linear(10, 3)         # 定义输出层
        self.out.weight.data.mul_(0.1)      # 初始化输出层权重

    def forward(self, x):
        x = self.fc1(x)                     # 输入张量经全连接层1传递
        x = torch.tanh(x)                   # 经tanh函数激活
        x = self.fc2(x)                     # 经全连接层2传递
        x = torch.tanh(x)                   # 经tanh函数激活
        x = self.fc3(x)                     # 经全连接层3传递
        x = torch.tanh(x)                   # 经tanh函数激活

        mu = self.out(x)                    # 经输出层传递得到输出张量
        logstd = torch.zeros_like(mu)       # 生成shape和mu相同的全0张量
        std = torch.exp(logstd)             # 生成shape和mu相同的全1张量

        std = std.mul(2)  # 动态设置std，标准差std越小越接近于均值，算法越倾向于利用，反之则越倾向于探索。

        return mu, std, logstd

    def choose_action(self, state):
        x = torch.FloatTensor(state)  # 输入状态数组转为张量 shape-torch.Size([80])
        # 网络前向推理 mu.shape-torch.Size([3]) std.shape-torch.Size([3])
        mu, std, _ = self.forward(x)
        action = torch.normal(mu, std).data.numpy()  # 按照给定的均值和方差生成输出张量的近似数据

        print("choose的原始action为", action)

        action[0] = np.clip(action[0], 2.4, 6)  # 按照[2.4, 6]的区间截取action[0]（初速度）
        action[1] = np.clip(action[1], -2, 2)  # 按照[-2, 2]的区间截取action[1]（横向偏移）
        # 按照[-3.14, 3.14]的区间截取action[2]（初始角速度）
        action[2] = np.clip(action[2], -3.14, 3.14)

        print("本次choose的均值为", mu)
        return action

# 创建Critic网络类继承自nn.Module


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(80, 128)       # 定义全连接层1
        self.fc2 = nn.Linear(128, 64)       # 定义全连接层2
        self.fc3 = nn.Linear(64, 10)        # 定义全连接层3
        self.out = nn.Linear(10, 1)         # 定义输出层
        self.out.weight.data.mul_(0.1)      # 初始化输出层权重

    def forward(self, x):
        x = self.fc1(x)                     # 输入张量经全连接层1传递
        x = torch.tanh(x)                   # 经tanh函数激活
        x = self.fc2(x)                     # 经全连接层2传递
        x = torch.tanh(x)                   # 经tanh函数激活
        x = self.fc3(x)                     # 经全连接层3传递
        x = torch.tanh(x)                   # 经tanh函数激活
        return self.out(x)                  # 经输出层传递得到输出张量