import os, time
import numpy as np
import torch
from aibot import AIRobot
from dqn import DQN
from util import get_infostate

#低速：在(2.4,2.7)之间以0.1为步长进行离散
slow = np.arange(2.4, 2.7, 0.1)
#中速：在(2.8,3.2)之间以0.05为步长进行离散
normal = np.arange(2.8, 3.2, 0.05)
#高速
fast = np.array([4,5,6])

#将低速、中速、高速三个数组连接起来
speed = np.concatenate((slow, normal, fast))

#横向偏移在(-2,2)之间以0.4为步长进行离散
deviation = np.arange(-2, 2, 0.4)
#角速度在(-3.14, 3.14)之间以0.628为步长进行离散
angspeed = np.arange(-3.14, 3.14, 0.628)

n = 0
#初始化动作列表
action_list = np.empty([1600, 3], dtype=float)
#遍历速度、横向偏移、角速度组合成各种动作
for i in speed:
    for j in deviation:
        for k in angspeed:
            action_list[n,] = [i, j, k]
            n += 1

class DQNRobot(AIRobot):
    def __init__(self, key, name, round_max):
        super().__init__(key, name)
        
        #初始化|加载先手模型
        self.dqn_init = DQN()
        self.init_model_file = 'model/DQN_init.pth'
        if os.path.exists(self.init_model_file):
            print("加载模型文件 %s" % (self.init_model_file))
            net_params = torch.load(self.init_model_file)
            self.dqn_init.eval_net.load_state_dict(net_params)
            self.dqn_init.target_net.load_state_dict(net_params)
        
        #初始化|加载后手模型
        self.dqn_dote = DQN()
        self.dote_model_file = 'model/DQN_dote.pth'  
        if os.path.exists(self.dote_model_file):
            print("加载模型文件 %s" % (self.dote_model_file))
            net_params = torch.load(self.dote_model_file)
            self.dqn_init.eval_net.load_state_dict(net_params)
            self.dqn_init.target_net.load_state_dict(net_params)
        
        self.learn_start = 100          # 学习起始局数
        self.round_max = round_max      # 最大训练局数
        self.log_file_name = 'log/DQN_log/traindata_' + time.strftime("%y%m%d_%H%M%S") + '.log' # 日志文件




    #根据当前比分获取奖励分数
    def get_reward(self, this_score):
        House_R = 1.830
        Stone_R = 0.145
        reward = this_score - self.last_score
        if (reward == 0):
            x = self.position[2*self.shot_num]
            y = self.position[2*self.shot_num+1]
            dist = self.get_dist(x, y)
            if dist < (House_R+Stone_R):
                reward = 1 - dist / (House_R+Stone_R)
        return reward
        
    #处理投掷状态消息
    def recv_setstate(self, msg_list):
        #当前完成投掷数
        self.shot_num = int(msg_list[0])
        #总对局数
        self.round_total = int(msg_list[2])
        
        #达到最大局数则退出训练
        if self.round_num == self.round_max:
            self.on_line = False
            return
        
        #每一局开始时将历史比分清零
        if (self.shot_num == 0):
            self.last_score = 0
        this_score = 0
            
        #根据先后手选取模型并设定当前选手第一壶是当局比赛的第几壶
        if self.player_is_init:
            first_shot = 0
            self.dqn = self.dqn_init
        else:
            first_shot = 1
            self.dqn = self.dqn_dote
            
        #当前选手第1壶投出前
        if self.shot_num == first_shot:
            init_score, self.s1 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.A = self.dqn.choose_action(self.s1)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第1壶投出后
        if self.shot_num == first_shot+1:
            init_score, s1_ = get_infostate(self.position)          # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
            reward = self.get_reward(this_score)                    # 获取动作奖励
            self.dqn.store_transition(self.s1, self.A, reward, s1_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                self.dqn.learn()  
        #当前选手第2壶投出前
        if self.shot_num == first_shot+2:
            init_score, self.s2 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.A = self.dqn.choose_action(self.s2)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第2壶投出后
        if self.shot_num == first_shot+3:
            init_score, s2_ = get_infostate(self.position)          # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
            reward = self.get_reward(this_score)                    # 获取动作奖励
            self.dqn.store_transition(self.s2, self.A, reward, s2_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                self.dqn.learn()
        #当前选手第3壶投出前
        if self.shot_num == first_shot+4:
            init_score, self.s3 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.A = self.dqn.choose_action(self.s3)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第3壶投出后
        if self.shot_num == first_shot+5:
            init_score, s3_ = get_infostate(self.position)          # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
            reward = self.get_reward(this_score)                    # 获取动作奖励
            self.dqn.store_transition(self.s3, self.A, reward, s3_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                self.dqn.learn()
        #当前选手第4壶投出前
        if self.shot_num == first_shot+6:
            init_score, self.s4 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.A = self.dqn.choose_action(self.s4)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第4壶投出后
        if self.shot_num == first_shot+7:
            init_score, s4_ = get_infostate(self.position)          # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
            reward = self.get_reward(this_score)                    # 获取动作奖励
            self.dqn.store_transition(self.s4, self.A, reward, s4_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                self.dqn.learn()
        #当前选手第5壶投出前
        if self.shot_num == first_shot+8:
            init_score, self.s5= get_infostate(self.position)       # 获取当前得分和状态描述
            self.A = self.dqn.choose_action(self.s5)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第5壶投出后
        if self.shot_num == first_shot+9:
            init_score, s5_ = get_infostate(self.position)          # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
            reward = self.get_reward(this_score)                    # 获取动作奖励
            self.dqn.store_transition(self.s5, self.A, reward, s5_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                self.dqn.learn()
        #当前选手第6壶投出前
        if self.shot_num == first_shot+10:
            init_score, self.s6 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.A = self.dqn.choose_action(self.s6)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第6壶投出后
        if self.shot_num == first_shot+11:
            init_score, s6_ = get_infostate(self.position)          # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
            reward = self.get_reward(this_score)                    # 获取动作奖励
            self.dqn.store_transition(self.s6, self.A, reward, s6_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                self.dqn.learn()
        #当前选手第7壶投出前
        if self.shot_num == first_shot+12:
            init_score, self.s7 = get_infostate(self.position)      # 获取当前得分和状态描述
            self.A = self.dqn.choose_action(self.s7)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
            self.last_score = (1-2*first_shot)*init_score           # 先手为正/后手为负
        #当前选手第7壶投出前
        if self.shot_num == first_shot+13:
            init_score, s7_ = get_infostate(self.position)          # 获取当前得分和状态描述
            this_score = (1-2*first_shot)*init_score                # 先手为正/后手为负
            reward = self.get_reward(this_score)                    # 获取动作奖励
            self.dqn.store_transition(self.s7, self.A, reward, s7_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                self.dqn.learn()
        #当前选手第8壶投出前
        if self.shot_num == first_shot+14:
            _, self.s8 = get_infostate(self.position)               # 获取当前状态描述
            self.A = self.dqn.choose_action(self.s8)                # 选择动作序号
            self.action = action_list[self.A]                       # 生成动作参数列表
        #当前选手第8壶投出后
        if self.shot_num == first_shot+15:
            _, self.s8_ = get_infostate(self.position)              # 获取当前得分和状态描述

        if self.shot_num == 16:
            if self.score > 0:                                      # 获取动作奖励
                reward = 5 * self.score
            else:
                reward = 0                
            self.dqn.store_transition(self.s8, self.A, reward, self.s8_) # 保存经验数据
            if self.dqn.memory_counter > self.learn_start:          # 记忆库满了就进行学习
                loss = self.dqn.learn()

            self.round_num += 1
            #将本局比分和当前loss值写入日志文件
            log_file = open(self.log_file_name, 'a+')
            log_file.write("score "+str(self.score)+" "+str(self.round_num)+"\n")
            if self.dqn.memory_counter > self.learn_start:
                log_file.write("loss "+str(loss)+" "+str(self.round_num)+"\n")
            log_file.close()
            
            #每隔50局存储一次模型
            if self.round_num % 3 == 0:
                net_params = self.dqn.eval_net.state_dict()
                if self.player_is_init:
                    torch.save(net_params, self.init_model_file)
                else:
                    torch.save(net_params, self.dote_model_file)
                print('============= Checkpoint Saved =============')

    def get_bestshot(self):
        return  "BESTSHOT " + str(self.action)[1:-1].replace(',', '')
    
if __name__ == '__main__':
    #根据数字冰壶服务器界面中给出的连接信息修改CONNECTKEY，注意这个数据每次启动都会改变。
    key = "test2023_2_1a5e475b-6a1c-499e-ae43-b67d72f2ab69"
    #初始化AI选手
    airobot = DQNRobot(key, "DQNRobot", round_max=10000)
    #启动AI选手处理和服务器的通讯
    airobot.recv_forever()