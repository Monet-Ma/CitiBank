
import os
import time
from collections import deque
import pickle
import threading
from Model_train.ddpg_learner import DDPG
from Model_train.models import Actor, Critic
from Model_train.memory import Memory
from Model_train.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
from gym import spaces, logger
from Model_train.trainEnv2 import  trainEnv
from Model_train.testEnv2 import testEnv
import numpy as np

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class WebAgent:
    def __init__(self, seed=None, action_space=None, observation_space=None,
                 network=None,
                 num_timesteps=None,
                 noise_type='ou_0.05',  #'adaptive-param_0.2' ou_0.2
                 normalize_returns=False,
                 normalize_observations=False,
                 critic_l2_reg=1e-2,
                 actor_lr=1e-3,
                 critic_lr=1e-4,
                 popart=False,
                 gamma=0.99,
                 clip_norm=None,
                 batch_size=32,  # per MPI worker
                 tau=0.001,
                 reward_scale=1.0,
                 runspace = None,
                 loadmodel=True,
                 buffer_size = 30000,
                 Max_episodes = 200,
                 Max_step = 800,
                 **network_kwargs):
        self.buffer_size = buffer_size
        self.runspace = runspace
        self.loadmodel = loadmodel
        self.seed = seed
        set_global_seeds(self.seed)
        self.observation_space = observation_space
        self.action_space = action_space
        self.network = network
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size
        # 模拟环境有关参数初始化
        #self.action_space = spaces.Discrete(31)

        # observation space
        # arrivalrate, queuing length
        high = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        low = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        high = np.array(high)
        low = np.array(low)
        self.observation_space = spaces.Box(low, high)
        # action space capacity
        low = np.array([0])
        high = np.array([1])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.lock = threading.Lock()
        self.NewSampleNumber = 0
        self.step = 32
        self.Max_episodes = Max_episodes
        self.Max_step = Max_step
        nb_actions = self.action_space.shape[-1]


        # assert (np.abs(self.action_space.low) == self.action_space.high).all()  # we assume symmetric actions 类似 [-1,1].

        self.memory = Memory(limit=int(self.buffer_size), action_shape=self.action_space.shape,
                        observation_shape=self.observation_space.shape)
        self.critic = Critic(network=self.network, **network_kwargs)
        self.actor = Actor(nb_actions, network=self.network, **network_kwargs)
        self.status = 0
        action_noise = None
        param_noise = None
        if noise_type is not None:
            for current_noise_type in noise_type.split(','):
                current_noise_type = current_noise_type.strip()
                if current_noise_type == 'none':
                    pass
                elif 'adaptive-param' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                         desired_action_stddev=float(stddev))
                elif 'normal' in current_noise_type:
                    _, stddev = current_noise_type.split('_')
                    action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
                    k = 1
                    while k < 100:
                        print('testing noise', action_noise())
                elif 'ou' in current_noise_type:
                    _, stddev = current_noise_type.split('_')

                    #print("stddev",stddev)
                    action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                                                sigma=float(stddev) * np.ones(nb_actions))
                    #k = 1
                    #while k < 100:
                    #    print(action_noise())
                    #    k = k + 1
                else:
                    raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        max_action = self.action_space.high
        #logger.info('scaling actions by {} before executing in env'.format(max_action))

        self.agent = DDPG(self.actor, self.critic, self.memory, self.observation_space.shape, self.action_space.shape,
                     gamma=gamma, tau=tau, normalize_returns=normalize_returns,
                     normalize_observations=normalize_observations,
                     batch_size=batch_size, action_noise=action_noise, param_noise=param_noise,
                     critic_l2_reg=critic_l2_reg,
                     actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
                     reward_scale=reward_scale,
                          runspace=self.runspace)
        #logger.info('Using agent with the following configuration:')
        #logger.info(str(self.agent.__dict__.items()))

        eval_episode_rewards_history = deque(maxlen=100)
        episode_rewards_history = deque(maxlen=100)
        self.sess = U.get_session()
        self.sess.as_default()    # add czc
        self.sess.graph.as_default()
        # Prepare everything.
        self.agent.initialize(self.sess)

        if self.loadmodel:
            self.agent.loadmodel()

        self.sess.graph.finalize()

        self.agent.reset()
        self.FinishTraining = False


    def store_sample(self, observation, action, reward, newobservation, finishstatus):


        if self.NewSampleNumber >= self.batch_size:
            return False
        #print('self.step=', self.step)

        self.lock.acquire()
        self.NewSampleNumber = self.NewSampleNumber + 1
        self.lock.release()


        observation = np.array(observation).reshape([-1,self.observation_space.shape[-1]])
        action = np.array(action).reshape([-1, self.action_space.shape[-1]])
        reward = np.array(reward).reshape([-1, 1])
        newobservation = np.array(newobservation).reshape([-1, self.observation_space.shape[-1]])
        finishstatus = np.array(finishstatus).reshape([-1, 1])
        with self.sess.as_default():  #跨线程时需要明确指定使用的session
            with self.sess.graph.as_default():
                self.agent.store_transition(observation, action, reward, newobservation,
                               finishstatus)  # the batched data will be unrolled in memory.py's append.

        return True

    def getAction(self, observation):
        action, q, _, _ = self.agent.step(observation, apply_noise=True, compute_Q=True)

        return action

    def learn(self,
              total_timesteps=None,
              nb_epochs=None,  # with default settings, perform 1M steps total
              nb_epoch_cycles=20,
              nb_rollout_steps=398,
              render=False,
              render_eval=False,
              nb_train_steps=50,  # per epoch cycle and MPI worker,
              nb_eval_steps=100,
              eval_env=None,
              param_noise_adaption_interval=50,
              **network_kwargs):

        #set_global_seeds(self.seed)
        self.FinishTraining = False
        if total_timesteps is not None:
            assert nb_epochs is None
            nb_epochs = int(total_timesteps) // (nb_epoch_cycles * nb_rollout_steps)
        else:
            nb_epochs = 500



        start_time = time.time()


        totaltrained_num = 0
        with self.sess.as_default():
            with self.sess.graph.as_default():
                while not self.FinishTraining:
                    time.sleep(1)
                    epoch_actor_losses = []
                    epoch_critic_losses = []
                    epoch_adaptive_distances = []

                    self.lock.acquire()
                    traintimes = self.NewSampleNumber
                    totaltrained_num = totaltrained_num + traintimes
                    self.step = self.step + traintimes
                    # 增加人工样本
                    if self.step >= 50:
                        #self.memory.generateBoundarySamples()
                        self.agent.reset() #将行为网络复制给putubed 行为网络
                        self.status = 1
                        for t_train in range(nb_train_steps):
                            if self.memory.nb_entries >= self.batch_size and t_train % param_noise_adaption_interval == 0:
                                distance = self.agent.adapt_param_noise()
                                epoch_adaptive_distances.append(distance)

                            cl, al = self.agent.train()
                            epoch_critic_losses.append(cl)
                            epoch_actor_losses.append(al)
                            self.agent.update_target_net()
                        #print('epoch_actor_losses', epoch_actor_losses)
                        #print('epoch_critic_losses', epoch_critic_losses)
                    self.status = 0
                    self.NewSampleNumber = 0
                    self.lock.release()
                    self.agent.savemodel()
                    self.FinishTraining = True

    def runthread(self):
        done = 1
        ep_reward = []
        env = trainEnv()
        for i in range(self.Max_episodes):
            ep_r = 0
            for j in range(self.Max_step):
                print("episode", i, "step", j)
                # if i*j <= 200:
                #     state1,state2 = env.step()
                #     print(state2[16])
                #     action = state2[16]
                #     act = np.array([[action/27000]])
                #     reward = env.calculate_r(state2,action)
                #     self.store_sample(state1,act,reward,state1,done)
                # else:
                state1,state2 = env.step()

                act= self.getAction(state2)
                action = act[0][0]
                print(action)
                reward = env.calculate_r(state2,action)
                print("reward",reward)
                self.store_sample(state2,act,reward,state2,done)
                ep_r = ep_r+reward
                if self.NewSampleNumber >= self.batch_size:
                    print("start learn")
                    self.learn()

            ep_reward.append(ep_r)
            print("////////////////////////////////")
            doc = open('out2.txt', 'a')
            print("/////////////////////////////////////////////", file=doc)
            print(ep_reward, file=doc)
            doc.close()

    def testthread(self):
        env = testEnv()
        ep_r = []
        expert = []
        actionlist = []
        for i in range(49):
            state1,state2 = env.step(i)
            act = self.getAction(state2)
            action = act[0][0]
            act1 = act[0][0]*env.maxprice
            exp = env.expert(state2)
            expert.append(exp)
            actionlist.append(act1)
            reward = env.calculate_r(state2,action)
            ep_r.append(reward)
        print("////////////////////////////////")
        doc = open('out.txt', 'a')
        print("/////////////////////////////////////////////", file=doc)
        print(expert,file=doc)
        print(actionlist,file=doc)
        print(ep_r, file=doc)
        doc.close()