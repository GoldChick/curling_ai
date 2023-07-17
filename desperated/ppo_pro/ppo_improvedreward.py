import numpy as np
import torch
import math


# 只在发射一个冰壶之后结算，如发射冰壶的global id为0，则调用此方法时currshot为1
# <param='firstshot'>是否是第一个射的，是为0，不是为1
# <param='currshot'>当前要射的冰壶id,此时场上只有这个id之前的冰壶的数据
# <param='oriposs'>发射之前的pos们，长度为32的数组(8*2*2)
# <param='currposs'>发射之后的pos们
def get_reward(firstshot: int, currshot: int, oriposs: np, currposs: np):
    rewards = 0
    # ori_me, ori_enemy = get_each_side(firstshot, oriposs)
    curr_me, curr_enemy = get_each_side(firstshot, currposs)
    # 处理之后，各方的冰壶数据形如[[1,2],[2,3],...]
    curr_index = round((currshot-1-firstshot)/2)  # 这个发射的冰壶在xxx_me中的index

    # sum2, _, nearest_id2 = how_many_point(ori_enemy)
    # sum4, _, nearest_id4 = how_many_point(curr_enemy)
    # # ------enemy------
    # out_flag = False
    # if sum4 < sum2:  # 击飞了对方的壶
    #     rewards += 3.0
    #     out_flag = True
    #     if sum4 == 0:  # 清空了对方的壶
    #         rewards += 9.0
    # elif sum4 > sum2:  # 怎么还帮对面???
    #     rewards -= 2.5
    # elif sum4 > 0:  # 没有改变enemy得分壶数量，于是看看质量
    #     ori_min = get_dist_by_point(ori_enemy[nearest_id2])
    #     curr_min = get_dist_by_point(curr_enemy[nearest_id4])
    #     if ori_min < curr_min:  # 打远了，也行吧
    #         rewards += 0.25
    #     elif ori_min > curr_min:  # 打近了???
    #         rewards -= 0.25

    # # ------me------
    # sum1, _, nearest_id1 = how_many_point(ori_me)
    # sum3, _, nearest_id3 = how_many_point(curr_me)

    # help_flag = False
    # if sum3 > sum1:  # 增加了有效壶
    #     rewards += 1.5
    #     help_flag = True
    # elif sum3 < sum1:  # 害群之马！
    #     rewards -= 2.0
    #     if sum3 == 0:  # 甚至直接清空了
    #         rewards -= 3.0
    # elif sum3 > 0:
    #     ori_min = get_dist_by_point(ori_me[nearest_id1])
    #     curr_min = get_dist_by_point(curr_me[nearest_id3])
    #     if ori_min > curr_min:  # 打近了
    #         rewards += 0.25
    #         help_flag = True
    #     elif ori_min < curr_min:
    #         rewards -= 0.25

    # if nearest_id3 == curr_index:  # 当前发射的壶成为了最近有效壶
    #     rewards += 1.5

    # ------curr------

    curr = curr_me[curr_index]
    x = curr[0]
    y = curr[1]
    if x > 0 and y > 0:
        # 靠近中间为1.2，靠近两边为0.4，为正则乘，为负则除
        center_modifer = 1.2-(abs(x-2.375)/2.96875)
        if (y <= 3.05):
            center_modifer *= (0.4+0.6*(math.pow(y/3.05, 2)))

        curr_dis2center = get_dist(x, y)
        R = 1.830+0.145
        if curr_dis2center <= 0.22:  # 核心区域
            rewards += 28.0*center_modifer
        if curr_dis2center <= R/6.0:  # 核心区域
            rewards += 16.1*center_modifer
        elif curr_dis2center <= R/5.0:  # 核心区域
            rewards += 11.4*center_modifer
        elif curr_dis2center <= R/3.0:  # 次核心区域
            rewards += 6.7*center_modifer
        elif curr_dis2center <= R*0.85:  # 进去了
            rewards += 2.35*center_modifer
        elif curr_dis2center <= R:
            rewards = 1.0*center_modifer
        elif curr_dis2center <= R*1.6:
            rewards = -2.0/center_modifer
        else:
            rewards = -4.0/center_modifer
        # else:
        #     if y < 3.05:  # 超过了有惩罚
        #         if not out_flag:
        #             rewards -= 0.3/center_modifer
        #         if not help_flag:
        #             rewards -= 0.15/center_modifer

        #     if curr_dis2center <= R*1.35:  # 还行吧
        #         rewards += 0.8*center_modifer
        #     elif curr_dis2center <= R*1.9:
        #         rewards += 0.2*center_modifer
        #     elif curr_dis2center <= R*2.25:
        #         rewards += 0.04*center_modifer
        #     elif curr_dis2center <= 5.2:  # 还行吧
        #         rewards -= 1.2/center_modifer
        #     else:  # 再远就不礼貌了
        #         rewards -= 3.2/center_modifer
    else:  # 很不幸飞出了场地
        rewards = -6.0
        # if not out_flag:  # 并且甚至还没有击飞别人的壶
        #     rewards -= 10.0
        # if not help_flag:
        #     rewards -= 5.0
    return rewards


# 游戏结束，通过最终比分获取
def get_reward_final(finalposs: np):
    pass


'''
TODO:暂时的预计reward设计:

发射壶到得分区中央,获得大量奖励
发射壶到得分区,获得奖励
发射距离稍近，不获得奖励
发射距离过近,获得惩罚

发射壶到得分区,并且击飞对方得分区壶,获得大量奖励

击飞对方得分区壶,使对方少得分,获得奖励
发射壶到中间防御,获得少量奖励
击偏对方得分区壶获得,少量奖励

没有击飞对方得分壶,并且超过得分区但不飞出场地,不获得奖励

'''


def get_each_side(firstshot: int, poss: int):
    t = torch.tensor(poss).reshape((8, 4))

    me = t.index_select(1, torch.tensor([0, 1]))
    enemy = t.index_select(1, torch.tensor([2, 3]))

    if firstshot == 1:  # 后手
        n = me
        me = enemy
        enemy = n

    return me, enemy
# 统计有多少得分，以及得分的壶的id(在队伍中的id),还有距离最近的壶的id


def how_many_point(positions: np):
    sum = 0
    list = []
    nearest_id = -1
    min_dis = 1e3
    for index in range(8):
        p = positions[index]
        if p[0] > 0 or p[1] > 0:
            dis = get_dist(p[0], p[1])
            if dis < min_dis:
                nearest_id = index
                min_dis = dis
            if dis <= 1.830+0.145:
                list.append(index)
                sum += 1
    return sum, list, nearest_id


def get_dist_by_point(p):
    return get_dist(p[0], p[1])

# <return/>某点距离某点的距离，默认第二个点为圆心


def get_dist(x, y, to_x=2.375, to_y=4.88):
    return math.sqrt((x-to_x)**2+(y-to_y)**2)
