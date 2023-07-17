import time
from aibot import AIRobot
import torch
from ppo_improved import PPO_PRO
import numpy as np
from util import get_dist
from ppo_improvedreward import get_reward


class PPOProRobot(AIRobot):
    # <param='round_max'>为正代表最大训练局，为负代表非训练模式
    def __init__(self, key, name='PPOProAiBot', round_max=10000, round_between_train=10, device='cpu'):
        super().__init__(key, name, show_msg=False)
        self.ppo = None
        self.round_max = round_max
        self.log_file_name = 'log/PPO_log/traindata_' + \
            time.strftime("%y%m%d_%H%M%S") + '.log'  # 日志文件

        self.device = device
        self.round_between_train = round_between_train
        self.rewards = 0.0
        self.transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'reward': [],
            'dones': [],
        }

    def recv_setstate(self, msg_list):
        self.shot_num = int(msg_list[0])
        self.round_total = int(msg_list[2])

        if self.round_num >= self.round_max:
            self.on_line = False
            return

        # 每一局开始时将历史比分清零
        if (self.shot_num == 0):
            self.last_score = 0

        if self.player_is_init:
            first_shot = 0
        else:
            first_shot = 1

        if self.ppo is None:
            self.ppo = PPO_PRO(first_shot, epochs=4)

        if self.shot_num >= first_shot and self.shot_num < 16:
            if (self.shot_num+first_shot) % 2 == 0:

                self.last_poss = self.position

                self.s = np.array(normalize(self.position)).flatten()

                init_score = get_simple_infostate(self.position)

                self.action = self.ppo.take_action(self.s)

                self.last_score = (1-2*first_shot) * \
                    init_score           # 先手为正/后手为负

            else:
                init_score = get_simple_infostate(self.position)

                reward = get_reward(first_shot, self.shot_num,
                                    self.last_poss, self.position)

                self.transition_dict['states'].append(normalize(self.position))
                self.transition_dict['actions'].append(self.action)
                self.transition_dict['next_states'].append(
                    normalize(self.position))
                self.transition_dict['reward'].append(reward)
                self.transition_dict['dones'].append(0)

                self.rewards += reward
        elif self.shot_num == 16:

            reward = 0
            if first_shot == 1:  # 后手的最终结算
                reward = get_reward(first_shot, self.shot_num,
                                    self.last_poss, self.position)

            reward_final = self.score*2.5
            reward += reward_final

            self.transition_dict['states'].append(normalize(self.position))
            self.transition_dict['actions'].append(self.action)
            self.transition_dict['next_states'].append(
                normalize(self.position))
            self.transition_dict['reward'].append(reward)
            self.transition_dict['dones'].append(1)

            self.rewards += reward

            self.round_num += 1

            log_file = open(self.log_file_name, 'a+')
            log_file.write('round:'+str(self.round_num)+" score:" +
                           str(self.score)+' rewards:'+str(self.rewards)+"\n")

            if (self.round_max > 0) and (self.round_num % self.round_between_train == 0):
                a_l, c_l = self.ppo.train_model(self.transition_dict)

                log_file.write('actor_loss:'+str(a_l)+'\n')
                log_file.write('critic_loss:'+str(c_l)+'\n')

                self.ppo.save_model()

                self.transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'reward': [],
                    'dones': [],
                }

            self.rewards = 0
            log_file.close()

    def get_bestshot(self):
        return "BESTSHOT " + str(self.action)[1:-1].replace(',', '')


def normalize(poss: list) -> list:
    x = [x/4.75 for x in poss]
    y = [y/10.61 for y in poss]
    x[1::2] = y[1::2]
    return x


def get_simple_infostate(position):
    House_R = 1.830
    Stone_R = 0.145

    init = np.empty([8], dtype=float)
    gote = np.empty([8], dtype=float)
    both = np.empty([16], dtype=float)
    # 计算双方冰壶到营垒圆心的距离
    for i in range(8):
        init[i] = get_dist(position[4 * i], position[4 * i + 1])
        both[2*i] = init[i]
        gote[i] = get_dist(position[4 * i + 2], position[4 * i + 3])
        both[2*i+1] = gote[i]
    # 找到距离圆心较远一方距离圆心最近的壶
    if min(init) <= min(gote):
        win = 0  # 先手得分
        d_std = min(gote)
    else:
        win = 1  # 后手得分
        d_std = min(init)

    init_score = 0  # 先手得分
    # 16个冰壶依次处理
    for i in range(16):
        x = position[2 * i]  # x坐标
        y = position[2 * i + 1]  # y坐标
        dist = both[i]  # 到营垒圆心的距离
        if (dist < d_std) and (dist < (House_R+Stone_R)) and ((i % 2) == win):
            # 如果是先手得分
            if win == 0:
                init_score = init_score + 1
            # 如果是后手得分
            else:
                init_score = init_score - 1

    # 返回先手得分和转为一维的状态数组
    return init_score


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    key = "test2023_2_1a5e475b-6a1c-499e-ae43-b67d72f2ab69"
    airobot = PPOProRobot(key, "PPOProAiBot", 10000, 4, device)
    # 启动AI选手处理和服务器的通讯
    airobot.recv_forever()
