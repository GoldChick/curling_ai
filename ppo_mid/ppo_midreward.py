import numpy as np


def get_reward(poss: np) -> int:
    return 0


def get_final_reward(poss: np, score: int) -> int:
    if score > 0:
        reward = 10 * score
    else:
        reward = 0
    return reward

    # # 根据当前比分获取奖励分数
    # def get_reward(self, this_score):
    #     House_R = 1.830
    #     Stone_R = 0.145
    #     # 以投壶后得分减去投壶前得分为奖励分
    #     reward = this_score - self.last_score
    #     if (reward == 0):
    #         x = self.position[2*self.shot_num]
    #         y = self.position[2*self.shot_num+1]
    #         dist = self.get_dist(x, y)
    #         # 对于大本营内的壶按照距离大本营圆心远近给奖励分
    #         if dist < (House_R+Stone_R):
    #             reward = 1 - dist / (House_R+Stone_R)
    #     return reward
