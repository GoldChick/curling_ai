import time
import torch

from collections import deque

from ppo_mid.ppo_mid import PPO_MID
from ppo_midreward import get_reward, get_final_reward
from ..aibot import AIRobot
from ..util import get_infostate


class PPOMIDRobot(AIRobot):
    def __init__(self, key: str, name='PPO_MIDairobot', train=True, round_between_train=2):
        super().__init__(key, name, show_msg=False)
        self.ppo = None
        self.train = train
        self.round_between_train = round_between_train

        self.memory = deque()
        self.round_max = 114514
        self.log_file_name = 'log/PPO_log/traindata_' + \
            time.strftime("%y%m%d_%H%M%S") + '.log'

    def recv_setstate(self, msg_list):
        self.shot_num = int(msg_list[0])
        # 总对局数
        self.round_total = int(msg_list[2])

        if self.round_num == self.round_max:
            self.on_line = False
            return

        if (self.shot_num == 0):
            self.last_score = 0
        this_score = 0

        if self.player_is_init:
            first_shot = 0
        else:
            first_shot = 1

        if self.ppo == None:
            self.ppo = PPO_MID(first_shot=first_shot)

        if self.shot_num >= first_shot and self.shot_num < 15+first_shot:
            if (self.shot_num-first_shot) % 2 == 0:
                init_score, self.s = get_infostate(self.position)
                self.action = self.ppo.choose_action(self.s)
                self.last_score = (1-2*first_shot) * \
                    init_score           # 先手为正/后手为负
            else:
                init_score, _ = get_infostate(self.position)
                this_score = (1-2*first_shot) * \
                    init_score                # 先手为正/后手为负
                self.memory.append(
                    [self.s, self.action, get_reward(this_score), 0])
        elif self.shot_num == 16:

            self.memory.append(
                [self.s, self.action, get_final_reward(self.position, self.score), 1])

            self.round_num += 1

            log_file = open(self.log_file_name, 'a+')
            log_file.write("score "+str(self.score) +
                           " "+str(self.round_num)+"\n")

            if self.train and self.round_num % self.round_between_train == 0:
                actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.ppo.LearningRate(self.round_num),
                                             weight_decay=0.0001)
                critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.ppo.LearningRate(self.round_num),
                                              weight_decay=0.0001)
                loss = self.ppo.train_model(self.memory, actor_opt, critic_opt)
                self.ppo.save_model()

                self.memory = deque()
                log_file.write("loss "+str(float(loss)) +
                               " "+str(self.round_num)+"\n")
            log_file.close()

    # 重载基类的方法，通过此处发送信息
    def get_bestshot(self):
        return "BESTSHOT " + str(self.action)[1:-1].replace(',', '')


if __name__ == '__main__':
    # 根据数字冰壶服务器界面中给出的连接信息修改CONNECTKEY，注意这个数据每次启动都会改变。
    print('测试启动PPOMIDAIBOT!')
    key = "test2023_2_1a5e475b-6a1c-499e-ae43-b67d72f2ab69"

    myrobot = PPOMIDRobot(key, "PPO_MIDaibot", round_max=10000)
    myrobot.recv_forever()
