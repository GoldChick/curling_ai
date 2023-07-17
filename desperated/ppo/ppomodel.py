import math
import torch
import numpy as np

BATCH_SIZE = 32                             # 批次尺寸
GAMMA = 0.9                                 # 奖励折扣因子
LAMDA = 0.9                                 # GAE算法的调整因子
EPSILON = 0.2                               # 截断调整因子

#生成动态学习率
def LearningRate(x):
    lr_start = 0.0001                       # 起始学习率
    lr_end = 0.0005                         # 终止学习率
    lr_decay = 20000                        # 学习率衰减因子
    return lr_end + (lr_start - lr_end) * math.exp(-1. * x / lr_decay)

# 输出连续动作的概率分布
def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

# 使用GAE方法计算优势函数
def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + GAMMA * running_returns * masks[t]
        running_tderror = rewards[t] + GAMMA * previous_value * masks[t] - values.data[t]
        running_advants = running_tderror + GAMMA * LAMDA * running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants
    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

# 替代损失函数
def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]
    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio

# 训练模型
def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory, dtype=object)

    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    values = critic(torch.Tensor(states))
    loss_list = []
    # step 1: get returns and GAEs and log probability of old policy
    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(np.array(actions)), mu, std, logstd)
    old_values = critic(torch.Tensor(states))
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    # step 2: get value loss and actor loss and update actor & critic
    for epoch in range(10):
        np.random.shuffle(arr)
        for i in range(n // BATCH_SIZE):
            batch_index = arr[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(np.array(actions))[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            values = critic(inputs)
            clipped_values = oldvalue_samples + torch.clamp(values - oldvalue_samples, -EPSILON, EPSILON)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            clipped_ratio = torch.clamp(ratio, 1.0 - EPSILON, 1.0 + EPSILON)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + critic_loss

            loss_list.append(loss)
            critic_optim.zero_grad()
            critic_loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

    return 0, sum(loss_list)/10
