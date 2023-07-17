from net import Net
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

BATCH_SIZE = 32            # 批次尺寸
LR = 0.0001                # 学习率
EPSILON = 0.95              # 最优选择动作百分比
GAMMA = 0.9                # 奖励折扣因子
TARGET_REPLACE_ITER = 25  # Q现实网络的更新频率
MEMORY_CAPACITY = 10000    # 记忆库大小

class DQN(object):
    def __init__(self):
        self.eval_net = Net()                         # 初始化评价网络
        self.target_net = Net()                       # 初始化目标网络
        self.sum_loss = 0                             # 初始化loss值
        self.learn_step_counter = 0                   # 用于目标网络更新计时
        self.memory_counter = 0                       # 记忆库计数
        self.memory = np.zeros((MEMORY_CAPACITY, 80 * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 设定torch的优化器为Adam
        self.loss_func = nn.MSELoss()                 # 以均方误差作为loss值
        self.min_loss = 10000

    #根据输入状态x返回输出动作的索引（而不是动作）
    def choose_action(self, x):
        # 选最优动作
        if np.random.uniform() < EPSILON:
            x = Variable(torch.FloatTensor(x))        # 将x转为pytorch变量 shape-torch.Size([80])
            actions_eval = self.eval_net(x)           # 评价网络前向推理 shape-torch.Size([1600])
            action = int(actions_eval.max(0)[1])      # 返回概率最大的动作索引
            print("选择了一次概率最大的动作索引"+str(action))
        # 选随机动作
        else:                                     
            action = np.random.randint(0, 1600)       # 在0-1600之间选一个随机整数
            print("选择了一个随机动作")
        return action                 

    #存储经验数据（s是输入状态，a是输出动作，r是奖励，s_是下一刻的状态）
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))         # 将输入元组的元素数组按水平方向进行叠加
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    #学习经验数据
    def learn(self):
        #每隔TARGET_REPLACE_ITER次更新目标网络参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print("更新了一次目标网络参数！")
        self.learn_step_counter += 1
        print("现在的learn_counter为"+str(self.learn_step_counter))
        #抽取记忆库中的批数据
        size = min(self.memory_counter, MEMORY_CAPACITY)
        sample_index = np.random.choice(size, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]                 # 抽取出来的数据 shape-(32, 162)
        b_s = Variable(torch.FloatTensor(b_memory[:, :80]))     # 输入数据的状态 shape-torch.Size([32, 80])
        b_a = Variable(torch.LongTensor(b_memory[:, 80:81]))    # 输入数据的动作 shape-torch.Size([32, 1])
        b_r = Variable(torch.FloatTensor(b_memory[:, 81:82]))   # 输入数据的奖励 shape-torch.Size([32, 1])
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -80:]))   # 输入数据的下一个状态 shape-torch.Size([32, 80])

        # 针对做过的动作 b_a 来选 q_eval 的值
        self.eval_net.train()                                   # 设定当前处于训练模式
        actions_eval = self.eval_net(b_s)                       # 评价网络前向推理 shape-torch.Size([32, 1600])
        q_eval = actions_eval.gather(1, b_a)                    # 选取第1维第b_a个数为评估Q值 shape-torch.Size([32, 1])

        max_next_q_values = torch.zeros(32, dtype=torch.float).unsqueeze(dim=1) #shape-torch.Size([32, 1])  
        for i in range(BATCH_SIZE):
            action_target = self.target_net(b_s_[i]).detach()   # 目标网络前向推理 shape-torch.Size([1600])
            max_next_q_values[i] = float(action_target.max(0)[0]) #返回输出张量中的最大值
        q_target = (b_r + GAMMA * max_next_q_values)            # 计算目标Q值 shape-torch.Size([32, 1])
        
        #计算loss值
        loss = self.loss_func(q_eval, q_target)
        loss_item = loss.item()
        if loss_item < self.min_loss:
            self.min_loss = loss_item
        
        self.optimizer.zero_grad()                              # 梯度清零
        loss.backward()                                         # 将loss进行反向传播并计算网络参数的梯度
        self.optimizer.step()                                   # 优化器进行更新
        return loss_item