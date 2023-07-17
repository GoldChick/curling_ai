from ppo_mid.ppo_midaibot import PPOMIDRobot

if __name__ == '__main__':
    print('正式运行PPOMIDAIBOT!')
    key = "test2023_2_1a5e475b-6a1c-499e-ae43-b67d72f2ab69"
    myrobot = PPOMIDRobot(key, "PPO_MIDaibot", train=False)
    myrobot.recv_forever()
