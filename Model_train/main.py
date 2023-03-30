from datetime import datetime
from threading import *
from baselines.common import models
import tensorflow as tf
from WebAgent import WebAgent
import sys

if __name__ == '__main__':
    #f = open("D:\Runspace\logs\python_" + str(0.1) + "DDPG.log", 'a')
    #sys.stdout = f
    #sys.stderr = f
    strext = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')
    #f = open("D:\Runspace\logs\python_" + strext + ".log", 'a')
    #sys.stdout = f
    #'ou_0.1' 'ou_0.05' 'ou_0.2'
    #实例化DeepQforWeb类 activation tf.nn.relu tf.tanh
    #是否采用sampling样本
    agent = WebAgent(network=models.mlp(num_layers=2, num_hidden=30, activation=tf.nn.relu),
    num_timesteps = 1e6, seed = 2, batch_size=32,actor_lr=0.001, critic_lr=0.0001, noise_type=None, runspace ='D:\Runspace\citibank',
                    loadmodel = True, buffer_size = 160000, critic_l2_reg=0.0001)
    #神经网络开始学习
    #Thread(target=agent.runthread).start()
    Thread(target=agent.testthread).start()
   #主客户端，接收obs和update_eps参数，给出神经网络的指导动作