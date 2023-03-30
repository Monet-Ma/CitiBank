import numpy as np

import random
import math

class trainEnv:
    def __init__(self):
        data_value = np.loadtxt(open("train_data.csv"), delimiter=",", skiprows=1)
        data_shape = data_value.shape
        print(data_shape)
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        new_data = np.zeros(shape=(data_rows, data_cols))
        for i in range(0, data_rows):
            for j in range(0, data_cols):
                data_col_min_values = min(data_value[:, j])
                data_col_max_values = max(data_value[:, j])
                new_data[i][j] = (data_value[i][j] - data_col_min_values) / (data_col_max_values - data_col_min_values)
            #new_data[i][16] = data_value[i][16]
        self.newdata = new_data
        #np.savetxt("new_traindata.csv",new_data,delimiter=',',fmt='%s')
        #print(self.newdata)

    def step(self):
        r = random.randint(0, 811)
        ssp_dress = float(self.newdata[r][0])
        sp_dress = float(self.newdata[r][1])
        dress = float(self.newdata[r][2])
        roles = float(self.newdata[r][3])
        retinue = float(self.newdata[r][4])
        ssp_goods = float(self.newdata[r][5])
        sp_goods = float(self.newdata[r][6])
        goods = float(self.newdata[r][7])
        head = float(self.newdata[r][8])
        head_frame = float(self.newdata[r][9])
        graffiti = float(self.newdata[r][10])
        waiting_action = float(self.newdata[r][11])
        pursue_action = float(self.newdata[r][12])
        music = float(self.newdata[r][13])
        score = float(self.newdata[r][14])
        platform = float(self.newdata[r][15])
        price = float(self.newdata[r][16])
        t_state1 = [ssp_dress, sp_dress, dress, roles, retinue, ssp_goods,sp_goods,goods,head,head_frame,graffiti,waiting_action,pursue_action,music,score,platform]
        t_state2 = np.hstack((t_state1,price))
        print("r",r)
        return t_state1,t_state2

    def expert(self,state):
        outcome = state[0]*0.15+state[1]*0.08+state[2]*0.06+state[3]*0.06+state[4]*0.05+state[5]*0.15+state[6]*0.08+state[7]*0.06+state[8]*0.05+state[9]*0.04+state[10]*0.06+state[11]*0.05+state[12]*0.06+state[13]*0.04+state[14]*0.08*state[16]*0.13
        return outcome


    def calculate_r(self,state,action):
        outcome = state[0]*0.15+state[1]*0.08+state[2]*0.06+state[3]*0.06+state[4]*0.05+state[5]*0.15+state[6]*0.08+state[7]*0.06+state[8]*0.05+state[9]*0.04+state[10]*0.06+state[11]*0.05+state[12]*0.06+state[13]*0.04+state[14]*0.08*state[16]*0.13
        reward = 0.1 - math.fabs(action-outcome)
        return reward

