import math
import numpy as np

#获取某一冰壶距离营垒圆心的距离
def get_dist(x, y):
    House_x = 2.375
    House_y = 4.88
    return math.sqrt((x-House_x)**2+(y-House_y)**2)

#根据冰壶比赛服务器发送来的场上冰壶位置坐标列表获取得分情况并生成信息状态数组
def get_infostate(position):
    House_R = 1.830
    Stone_R = 0.145

    init = np.empty([8], dtype=float)
    gote = np.empty([8], dtype=float)
    both = np.empty([16], dtype=float)
    #计算双方冰壶到营垒圆心的距离
    for i in range(8):
        init[i] = get_dist(position[4 * i], position[4 * i + 1])
        both[2*i] = init[i] 
        gote[i] = get_dist(position[4 * i + 2], position[4 * i + 3])
        both[2*i+1] = gote[i]
    #找到距离圆心较远一方距离圆心最近的壶
    if min(init) <= min(gote):
        win = 0                     #先手得分
        d_std = min(gote)
    else:
        win = 1                     #后手得分
        d_std = min(init)
    
    infostate = []  #状态数组
    init_score = 0  #先手得分
    #16个冰壶依次处理
    for i in range(16):
        x = position[2 * i]         #x坐标
        y = position[2 * i + 1]     #y坐标
        dist = both[i]              #到营垒圆心的距离
        sn = i % 2 + 1              #投掷顺序
        if (dist < d_std) and (dist < (House_R+Stone_R)) and ((i%2) == win):
            valid = 1               #是有效得分壶
            #如果是先手得分
            if win == 0:
                init_score = init_score + 1
            #如果是后手得分
            else:
                init_score = init_score - 1
        else:
            valid = 0               #不是有效得分壶
        #仅添加有效壶
        if x!=0 or y!=0:
            infostate.append([x, y, dist, sn, valid])
    #按dist升序排列
    infostate = sorted(infostate, key=lambda x:x[2])
    
    #无效壶补0
    for i in range(16-len(infostate)):
        infostate.append([0,0,0,0,0])

    #返回先手得分和转为一维的状态数组
    return init_score, np.array(infostate).flatten()