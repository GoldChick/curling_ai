import time
import os
import math
from aibot import AIRobot
from ppo import Actor, Critic
import torch
from collections import deque
from util import get_infostate
from ppomodel import LearningRate, train_model


class PPORobot(AIRobot):
    def __init__(self, key, name, round_max):
        super().__init__(key, name, show_msg=False)

        # 初始化并加载先手actor模型
        self.init_actor = Actor()
        self.init_actor_file = 'model/PPO_init_actor.pth'
        if os.path.exists(self.init_actor_file):
            print("加载模型文件 %s" % (self.init_actor_file))
            self.init_actor.load_state_dict(torch.load(self.init_actor_file))

        # 初始化并加载先手critic模型
        self.init_critic = Critic()
        self.init_critic_file = 'model/PPO_init_critic.pth'
        if os.path.exists(self.init_critic_file):
            print("加载模型文件 %s" % (self.init_critic_file))
            self.init_critic.load_state_dict(torch.load(self.init_critic_file))

        # 初始化并加载后手actor模型
        self.dote_actor = Actor()
        self.dote_actor_file = 'model/PPO_dote_actor.pth'
        if os.path.exists(self.dote_actor_file):
            print("加载模型文件 %s" % (self.dote_actor_file))
            self.dote_actor.load_state_dict(torch.load(self.dote_actor_file))

        # 初始化并加载后手critic模型
        self.dote_critic = Critic()
        self.dote_critic_file = 'model/PPO_dote_critic.pth'
        if os.path.exists(self.dote_critic_file):
            print("加载模型文件 %s" % (self.dote_critic_file))
            self.dote_critic.load_state_dict(torch.load(self.dote_critic_file))

        self.memory = deque()               # 清空经验数据
        self.round_max = round_max          # 最多训练局数
        self.log_file_name = 'log/PPO_log/traindata_' + \
            time.strftime("%y%m%d_%H%M%S") + '.log'  # 日志文件

    # 根据当前比分获取奖励分数
    def get_reward(self, this_score):
        House_R = 1.830
        Stone_R = 0.145
        # 以投壶后得分减去投壶前得分为奖励分
        reward = this_score - self.last_score
        print("这个是不是有问题:", reward)
        x = self.position[2*self.shot_num-2]
        y = self.position[2*self.shot_num-1]

        dist = self.get_dist(x, y)

        if dist > 2.7:
            reward = reward + 1.5 - 2**(dist-3.7)
        # 对于大本营内的壶按照距离大本营圆心远近给更多的奖励分
        elif dist < (House_R+Stone_R):
            if dist < (House_R+Stone_R)/2:
                reward = reward + 17.0 - 16.0 * \
                    math.pow((dist / (House_R+Stone_R)), 4)
            else:
                reward = reward + 5.0 - 4.0 * \
                    math.pow((dist / (House_R+Stone_R)), 2)
        else:
            reward = reward + 1.0

        if reward == 0:  # 如果此次没有提高自己得分还出界了
            if x == 0 and y == 0:
                reward = -10
        print("计算出的奖励为"+str(reward*100))
        return reward*100

    # 处理投掷状态消息
    def recv_setstate(self, msg_list):
        # 当前完成投掷数
        self.shot_num = int(msg_list[0])
        # 总对局数
        self.round_total = int(msg_list[2])

        # 达到最大局数则退出训练
        if self.round_num == self.round_max:
            self.on_line = False
            return

        # 每一局开始时将历史比分清零
        if (self.shot_num == 0):
            self.last_score = 0
        this_score = 0

        # 根据先后手选取模型并设定当前选手第一壶是当局比赛的第几壶
        if self.player_is_init:
            first_shot = 0
            self.actor = self.init_actor
            self.critic = self.init_critic
        else:
            first_shot = 1
            self.actor = self.dote_actor
            self.critic = self.dote_critic

        if self.shot_num < 16:
            if (self.shot_num-first_shot) % 2 == 0:
                init_score, self.s = get_infostate(
                    self.position)      # 获取当前得分和状态描述
                self.action = self.actor.choose_action(
                    self.s)         # 根据状态获取对应的动作参数列表
                self.last_score = (1-2*first_shot) * \
                    init_score           # 先手为正/后手为负
            else:
                init_score, self.new_s = get_infostate(
                    self.position)            # 获取当前得分和状态描述
                this_score = (1-2*first_shot) * \
                    init_score                # 先手为正/后手为负
                reward = self.get_reward(
                    this_score)                    # 获取动作奖励
                self.memory.append(
                    [self.s, self.new_s, self.action, reward, 1])   # 保存经验数据
        else:
            if self.score > 0:
                reward = 100 * self.score                            # 获取动作奖励
            else:
                reward = 30 * self.score
            _, self.new_s = get_infostate(
                self.position)
            self.memory.append(
                [self.s, self.new_s, self.action, reward, 0])   # 保存经验数据

            self.round_num += 1
            # 如果处于训练模式且有12局数据待训练
            if (self.round_max > 0) and (self.round_num % 12 == 0):
                # 训练模型
                actor_optim = torch.optim.Adam(
                    self.actor.parameters(), lr=LearningRate(self.round_num))
                critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LearningRate(self.round_num),
                                                weight_decay=0.0001)
                self.actor.train(), self.critic.train()
                _, loss = train_model(
                    self.actor, self.critic, self.memory, actor_optim, critic_optim)
                # 保存模型
                if self.player_is_init:
                    torch.save(self.actor.state_dict(), self.init_actor_file)
                    torch.save(self.critic.state_dict(), self.init_critic_file)
                else:
                    torch.save(self.actor.state_dict(), self.dote_actor_file)
                    torch.save(self.critic.state_dict(), self.dote_critic_file)
                print('============= Checkpoint Saved =============')
                # 清空训练数据
                self.memory = deque()

            # 将本局比分和当前loss值写入日志文件
            # log_file = open(self.log_file_name, 'a+')
            # log_file.write("score "+str(self.score) +
            #                " "+str(self.round_num)+"\n")
            # if self.round_num % 12 == 0:
            #     log_file.write("loss "+str(float(loss)) +
            #                    " "+str(self.round_num)+"\n")
            # log_file.close()

    def start_train(self):
        pass
    def get_bestshot(self):
        return "BESTSHOT " + str(self.action)[1:-1].replace(',', '')


if __name__ == '__main__':
    # 根据数字冰壶服务器界面中给出的连接信息修改CONNECTKEY，注意这个数据每次启动都会改变。
    key = "test2023_2_1a5e475b-6a1c-499e-ae43-b67d72f2ab69"
    # 初始化AI选手
    airobot = PPORobot(key, "PPORobot", round_max=10000)
    # 启动AI选手处理和服务器的通讯
    airobot.recv_forever()
