import torch.nn as nn
import torch
import numpy as np


class Actor(nn.Module):
    def __init__(self, input_num=80, output_num=1):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_num, 128)       # 定义全连接层1
        self.fc2 = nn.Linear(128, 64)       # 定义全连接层2
        self.fc3 = nn.Linear(64, 10)        # 定义全连接层3
        self.out = nn.Linear(10, output_num)         # 定义输出层
        self.out.weight.data.mul_(0.1)      # 初始化输出层权重

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)

        mu = self.out(x)
        logstd = torch.zeros_like(mu)       # 生成shape和mu相同的全0张量
        std = torch.exp(logstd)             # 生成shape和mu相同的全1张量
        return mu, std, logstd

    def choose_action(self, state):
        x = torch.FloatTensor(state)
        # 网络前向推理 mu.shape-torch.Size([3]) std.shape-torch.Size([3])
        mu, std, _ = self.forward(x)
        action = torch.normal(mu, std).data.numpy()  # 按照给定的均值和方差生成输出张量的近似数据

        action[0] = np.clip(action[0], 2.4, 6)  # 按照[2.4, 6]的区间截取action[0]（初速度）
        action[1] = np.clip(action[1], -2, 2)  # 按照[-2, 2]的区间截取action[1]（横向偏移）
        # 按照[-3.14, 3.14]的区间截取action[2]（初始角速度）
        action[2] = np.clip(action[2], -3.14, 3.14)
        return action


class Critic(nn.Module):
    def __init__(self, input_num=80, output_num=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_num, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.out = nn.Linear(10, output_num)
        self.out.weight.data.mul_(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return self.out(x)
