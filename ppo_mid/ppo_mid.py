import math
import time
import torch
import numpy as np
import os
from torch.nn import functional as F
from ppo_midmodel import Actor, Critic

BATCH_SIZE = 32                             # 批次尺寸
GAMMA = 0.9                                 # 折扣因子
LAMDA = 0.9                                 # GAE算法的调整因子
CLIP = 0.1                               # 截断调整因子

INIT_ACTOR_FILE = 'model/PPO_init_actor.pth'
INIT_CRITIC_FILE = 'model/PPO_init_critic.pth'
DOTE_ACTOR_FILE = 'model/PPO_dote_actor.pth'
DOTE_CRITIC_FILE = 'model/PPO_dote_critic.pth'


class PPO_MID:
    def __init__(self, first_shot=0,  epochs=8, clip=0.2, device='cpu'):
        self.actor = Actor()
        self.critic = Critic()

        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.clip = clip  # PPO中截断范围的参数
        self.device = device
        self.action_file_name = 'log/actiondata_' + \
            time.strftime("%y%m%d_%H%M%S") + '.log'  # 日志文件

        if first_shot == 0:
            self.actor_file = INIT_ACTOR_FILE
            self.critic_file = INIT_CRITIC_FILE
        elif first_shot == 1:
            self.actor_file = DOTE_ACTOR_FILE
            self.critic_file = DOTE_CRITIC_FILE
        else:
            print('first_shot初始值错误!可能会造成某些问题!')
            assert (1 == 0)

        self.first_shot = first_shot
        if os.path.exists(self.actor_file):
            print("加载模型文件 %s 成功" % (self.actor_file))
            self.actor.load_state_dict(torch.load(self.actor_file))
        else:
            print('不存在已有的actor!将会重新创建')

        if os.path.exists(self.critic_file):
            print("加载模型文件 %s 成功" % (self.critic_file))
            self.critic.load_state_dict(torch.load(self.critic_file))
        else:
            print('不存在已有的critic!将会重新创建')

    def save_model(self):
        torch.save(self.actor.state_dict(), self.actor_file)
        torch.save(self.critic.state_dict(), self.critic_file)
        print('============= Checkpoint Saved =============')

    def get_learning_rate(self, round: int) -> int:
        lr_start = 0.0001                       # 起始学习率
        lr_end = 0.0005                         # 终止学习率
        lr_decay = 20000                        # 学习率衰减因子
        return lr_end + (lr_start - lr_end) * math.exp(-1. * round / lr_decay)

    def choose_action(self, state):
        return self.actor(state)

    # <param='values'>self.critic(states)

    def get_gae(self, rewards: torch.Tensor, dones: torch.Tensor, values) -> torch.Tensor:
        rewards = torch.Tensor(rewards)
        dones = torch.Tensor(dones)
        returns = torch.zeros_like(rewards)
        advantage = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0

        running_adv = 0.0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + GAMMA * \
                running_returns * (1-dones[t])

            running_tderror = rewards[t] + GAMMA * \
                previous_value * (1-dones[t]) - values.data[t]

            running_adv = running_tderror + GAMMA * \
                LAMDA * running_adv * (1-dones[t])

            returns[t] = running_returns
            previous_value = values.data[t]
            advantage[t] = running_adv

        advantage = (advantage - advantage.mean()) / advantage.std()
        return returns, advantage

    def getL1_ratio(self, advants, states, old_policy, actions, index):
        mu, std, logstd = self.actor(torch.Tensor(states))
        new_policy = log_density(actions, mu, std, logstd)
        old_policy = old_policy[index]
        ratio = torch.exp(new_policy - old_policy)
        L1 = ratio * advants
        return L1, ratio

    def train_model(self, memory, actor_optim: torch.optim.Adam, critic_optim: torch.optim.Adam):
        self.actor.train()
        self.critic.train()  # TODO:这个干嘛的

        memory = np.array(memory, dtype=object)
        states = np.vstack(memory[:, 0])
        actions = np.array(list(memory[:, 1]))
        rewards = list(memory[:, 2])
        dones = list(memory[:, 3])

        with torch.no_grad():
            old_values = self.critic(torch.Tensor(states))
            # still dont know what 'returns' is
            returns, advants = self.get_gae(
                rewards, dones, old_values)

            mu, std, logstd = self.actor(torch.Tensor(states))
            old_log_prob = log_density(torch.Tensor(actions), mu, std, logstd)

            states_num = len(states)
            states_num_array = np.arange(states_num)
            loss_list = []  # 最终返回loss总和

        for _ in range(self.epochs):
            np.random.shuffle(states_num_array)
            for i in range(states_num // BATCH_SIZE):
                # get samples
                batch_index = states_num_array[BATCH_SIZE *
                                               i: BATCH_SIZE * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = torch.Tensor(states)[batch_index]
                returns_samples = returns.unsqueeze(1)[batch_index]
                advants_samples = advants.unsqueeze(1)[batch_index]
                actions_samples = torch.Tensor(actions)[batch_index]
                oldvalue_samples = old_values[batch_index].detach()
                # critic
                values = self.critic(inputs)
                clipped_values = oldvalue_samples + \
                    torch.clamp(values - oldvalue_samples, -CLIP, CLIP)
                critic_loss1 = F.mse_loss(clipped_values, returns_samples)
                critic_loss2 = F.mse_loss(values, returns_samples)
                critic_loss = torch.max(critic_loss1, critic_loss2).mean()
                # actor
                L1, ratio = self.getL1_ratio(advants_samples, inputs,
                                             old_log_prob.detach(), actions_samples,
                                             batch_index)
                clip = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
                L2 = clip * advants_samples
                actor_loss = -torch.min(L1, L2).mean()
                # output
                loss = actor_loss + critic_loss
                loss_list.append(loss)
                # update
                actor_optim.zero_grad()
                critic_optim.zero_grad()
                actor_loss.backward()
                critic_loss.backward(retain_graph=True)
                actor_optim.step()
                critic_optim.step()

        return sum(loss_list)/10


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * \
        math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)
