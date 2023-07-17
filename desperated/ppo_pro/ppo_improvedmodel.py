import torch.nn as nn
from torch.nn import functional as F


class PolicyNet(nn.Module):
    def __init__(self, n_states=32, n_hiddens=128, n_actions=3):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc_mu = nn.Linear(n_hiddens, n_actions)
        self.fc_std = nn.Linear(n_hiddens, n_actions)

        self.fc_mu.weight.data.mul_(0.5)
        self.fc_std.weight.data.mul_(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        mu = self.fc_mu(x)
        mu[0][0] = 3.0+0.5*F.tanh(mu[0][0])  # [2.0,4.0]
        mu[0][1] = 2.23*F.tanh(mu[0][1])
        mu[0][2] = 0.314*F.tanh(mu[0][2])  # 限制不旋转那么多

        std = self.fc_std(x)
        std = F.sigmoid(std)
        return mu, std


class ValueNet(nn.Module):
    def __init__(self, n_states=32, n_hiddens=128):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.out = nn.Linear(n_hiddens, 1)
        self.out.weight.data.mul_(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.out(x)
