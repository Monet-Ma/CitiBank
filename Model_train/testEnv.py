import numpy as np
#单位线路长度造价水平权重	1类项目	1类线路类型	E1*0.5+H1*0.2+C1*0.1+I1*0.2
#                       1类项目	2类线路类型	E1*0.4+H1*0.2+C1*0.2+I1*0.2


class testEnv:
    def __init__(self):
        data_value = np.loadtxt(open("test_data.csv"), delimiter=",", skiprows=1)
        data_shape = data_value.shape
        print(data_shape)
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        new_data = np.zeros(shape=(data_rows, data_cols))
        for i in range(0, data_rows, 1):
            for j in range(0, data_cols, 1):
                data_col_min_values = min(data_value[:, j])
                data_col_max_values = max(data_value[:, j])
                new_data[i][j] = (data_value[i][j] - data_col_min_values) / (data_col_max_values - data_col_min_values)
        self.newdata = new_data
        #np.savetxt("new_traindata.csv",new_data,delimiter=',',fmt='%s')

    def step(self,j):
        type = float(self.newdata[j][0])
        invest = float(self.newdata[j][1])
        capacitance = float(self.newdata[j][2])
        n_wire_length = float(self.newdata[j][3])
        n_transmission_capacity = float(self.newdata[j][4])
        wire_length = float(self.newdata[j][5])
        voltage = float(self.newdata[j][6])
        t_state1 = [invest, capacitance, n_wire_length, n_transmission_capacity, wire_length, voltage]
        t_state2 = np.hstack((type,t_state1))
        return t_state1,t_state2

    def output(self,state):
        output1 = 0
        if state[0] == 0:
            output1 = state[3] * 0.5 + state[5] * 0.2 + state[1] * 0.1 + state[6] * 0.2
        if state[0] == 1:
            output1 = state[3] * 0.4 + state[5] * 0.2 + state[1] * 0.2 + state[6] * 0.2

        return output1

    def calculate_r(self,state,action):
        output1 = 0
        if state[0]==0:
            output1 = state[3]*0.5+state[5]*0.2+state[1]*0.1+state[6]*0.2
        if state[0]==1:
            output1 = state[3]*0.4+state[5]*0.2+state[1]*0.2+state[6]*0.2
        reward = 0.02-((action-output1)*(action-output1))
        return reward