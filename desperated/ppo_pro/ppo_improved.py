import time
import torch
from torch.nn import functional as F
import numpy as np
import os
from ppo_improvedmodel import PolicyNet, ValueNet


class PPO_PRO:
    # <param='first_shot'>为0代表是先手，读取先手的pth文件
    def __init__(self, first_shot=0, actor_lr=2e-5, critic_lr=2e-4, lmbda=0.95, epochs=8, clip=0.2, gamma=0.99, device='cpu'):
        self.actor = PolicyNet().to(device)
        self.critic = ValueNet().to(device)
        self.old_actor = PolicyNet().to(device)
        self.old_critic = ValueNet().to(device)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.clip = clip  # PPO中截断范围的参数
        self.device = device
        self.action_file_name = 'log/PPO_log/actiondata_' + \
            time.strftime("%y%m%d_%H%M%S") + '.log'  # 日志文件

        init_actor_file = 'model/PPO_init_actor.pth'
        init_critic_file = 'model/PPO_init_critic.pth'
        dote_actor_file = 'model/PPO_dote_actor.pth'
        dote_critic_file = 'model/PPO_dote_critic.pth'

        if first_shot == 0:
            self.actor_file = init_actor_file
            self.critic_file = init_critic_file
        elif first_shot == 1:
            self.actor_file = dote_actor_file
            self.critic_file = dote_critic_file
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

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)
        # 动作选择

    # <return/>返回[1.0,2.0,3.0]这样的数组
    def take_action(self, state):
        with torch.no_grad():
            state = torch.tensor(state[np.newaxis, :],
                                 dtype=torch.float).to(self.device)
            mu, std = self.actor(state)
            action_dict = torch.distributions.Normal(mu, std)
            action = action_dict.sample()
            # print('选择的action为:', action)
            log_file = open(self.action_file_name, 'a+')
            log_file.write('mu:'+str(mu)+'\n')
            log_file.write('std:'+str(std)+'\n')
            log_file.write('action:'+str(action)+'\n\n')
            log_file.close()

        return action[0].numpy()  # 返回动作值

    # <return/>两个loss
    def train_model(self, transition_dict):
        # 提取数据集
        states = torch.tensor(
            transition_dict['states'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'], dtype=torch.float).to(self.device)

        actions = torch.tensor(transition_dict['actions']).to(
            self.device).view(-1, 3)

        rewards = torch.tensor(transition_dict['reward'], dtype=torch.float).to(
            self.device).view(-1, 1)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(
            self.device).view(-1, 1)

        with torch.no_grad():
            next_states_target = self.critic(next_states)
            td_target = rewards + self.gamma * \
                next_states_target * (1-dones)  # loss_value

            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(
                mu, std)
            old_log_prob = action_dists.log_prob(actions)  # loss_policy

            td_delta = td_target-self.critic(states)

            td_delta = td_delta.cpu().detach().numpy()

            advantage = 0.0
            advantage_list = []
            for td in td_delta[::-1]:
                advantage = self.gamma * self.lmbda * advantage + td
                advantage_list.append(advantage)
            advantage_list.reverse()

            advantage = torch.tensor(
                advantage_list, dtype=torch.float).reshape(-1, 1).to(self.device)

        actor_loss = 0
        critic_loss = 0

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_prob_new = action_dists.log_prob(actions)

            ratio = torch.exp(log_prob_new - old_log_prob)

            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantage

            actor_loss = -torch.min(L1, L2).mean()

            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss, critic_loss

    def save_model(self):
        torch.save(self.actor.state_dict(), self.actor_file)
        torch.save(self.critic.state_dict(), self.critic_file)
