# 冰壶，启动！

## PPO_MID
### 采取的方案
预计采用折中方案2，即手动编写有限个策略，再训练输出离散值的神经网络
### 使用的包 
python使用python3.11.1,包的版本详见[requirements.txt](./requirements.txt)
### 如何启动<font color=red>(注：由于没有修改好网络，暂时不应使用)</font>
运行[ppo_midaibot.py](./ppo_mid/ppo_midaibot.py)可以开始训练，模型保存在[同目录下的model文件夹](./ppo_mid/model/)中

运行[main.py](./main.py)可以启动但是不训练
### 使用的强化学习算法
采用PPO算法(截至目前距离完成进度相差还很大)

### <font color=00ffff>目前施工状态：
正在编写各种策略，模型还是复制粘贴过来的输出连续值的神经网络</font>
## <font color=ff00ff>QUESTIONS</font>
### 想使用其他算法
可以在根目录下创建其他文件夹，如./dqn/之类的
### 想在原有基础上改动
直接修改即可，但是需要在某些地方（如提交git时的说明、或者直接代码内部）表明改动、增加的内容
<hr>


