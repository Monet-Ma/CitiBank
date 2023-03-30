from baselines.common import models
import tensorflow as tf
from Model_train.WebAgent import WebAgent
class Predict:
    def __init__(self,account_item):
        self.agent = WebAgent(network=models.mlp(num_layers=10, num_hidden=60, activation=tf.nn.relu),
                         num_timesteps=1e6, seed=2, batch_size=64, actor_lr=0.01, critic_lr=0.001,
                         noise_type='adaptive-param_0.1', runspace='D:\Runspace\ddpg',
                         loadmodel=True, buffer_size=200000, critic_l2_reg=0.0001)

        self.state = account_item


    def prdict(self):
        act = self.agent.getAction(self.state)
        action = act[0][0] * 64000
        return action