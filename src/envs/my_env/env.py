from .satellite import Satellite
from .dataStream import DataStream
import numpy as np
import random
import gym
from gym import spaces

import sys
import os

# 添加父目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from multiagentenv import MultiAgentEnv




class myEnv(MultiAgentEnv):
    # 50个卫星 5个源 1个end
    def __init__(
        self,
        episode_limit=1000,
        n_agents=5,
        satellite_num=50,
        **kwargs,
        ):
        super(myEnv, self).__init__()
        self.time = 0 # 时隙的轮次
        self.timeSlot = 1 # 每个时隙的大小
        self.episode_limit = episode_limit # 最大时隙
        self.n_agents = n_agents
        self.n_actions = 7
        self.satellite_num = satellite_num
        self.adj = self.generateAdj()  # 邻接矩阵
        self.satellite_list = []
        self.agent_list = []
        self.compression_ratio_list = [0, 0.05, 0.1] # 图像压缩率的列表
        self.alpha = 0.5  # 传输时间和数据量之间的权重系数
        self.observation_space = 1 + 1 + 4 + 4 + 1  # 状态维度，数据量-1 + 所在卫星ID-1 + 四个邻居卫星的带宽-4 + 四个邻居卫星的ID-4 + 目的地-1
        # self.action_space = 7
        # self.action_space = [spaces.Discrete([7]) for _ in range(self.n_agents)]
        # self.observation_space = spaces
        self.end = random.randint(self.n_agents, self.satellite_num - 1)  # 前n_agents个卫星是EO卫星，从后面的id中随机生成一个终点

        for i in range(self.satellite_num):  # 生成所有的卫星对象
            neighbor_ids = self.adj[i]
            neighbor_bandwidths = []
            for j in range(4):
                neighbor_bandwidths.append(random.randint(5, 10))
            if (i >= 0) and (i < self.n_agents):
                self.satellite_list.append(
                    Satellite(i, neighbor_ids, neighbor_bandwidths, 0))
            elif i == self.end:
                self.satellite_list.append(
                    Satellite(i, neighbor_ids, neighbor_bandwidths, 2))
            else:
                self.satellite_list.append(
                    Satellite(i, neighbor_ids, neighbor_bandwidths, 1))

        for i in range(self.n_agents):  # 生成所有的agent对象
            data_amount = random.randint(50, 100)
            self.agent_list.append(DataStream(i, i, data_amount))

    def take_action(self, action):  # 参数的action是从policy中sample出来的策略
        # print(action)
        reward = np.zeros(self.n_agents)

        for i in range(self.n_agents):  # 更新agent的下一跳目的和数据量
            
            if self.agent_list[i].isTransmitting or self.agent_list[i].curr_satellite_id == self.end:
                continue
            i_agent_action = action[i]
            # temp_list = []  # 转发动作和丢弃动作的下标
            # for j, num in enumerate(range(int(i_agent_action))):
            #     if num == 1:
            #         temp_list.append(j)
            # print(temp_list)
            
            if i_agent_action < 4:
                target = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor[i_agent_action]
                self.agent_list[i].next_satellite_id = target  # 修改下一跳目的
                # print(f"agent {i} transfer to {target}, end is {self.end}, time_step is {self.time}")
            else:
                self.agent_list[i].data_amount *= (1 - self.compression_ratio_list[i_agent_action - 4])  # 修改数据量

        map = {}
        for i in range(self.n_agents):  # 获取每个agent能分得的带宽
            if self.agent_list[i].isTransmitting or self.agent_list[i].curr_satellite_id == self.end:
                continue
            x, y = self.agent_list[i].curr_satellite_id, self.agent_list[i].next_satellite_id
            key = str(x) + "-" + str(y)
            if key in map:
                map[key] += self.agent_list[i].data_amount
            else:
                map[key] = self.agent_list[i].data_amount

        for i in range(self.n_agents):  # 修改到达时间和isTransmitting状态，以及计算reward
            if self.agent_list[i].isTransmitting or self.agent_list[i].curr_satellite_id == self.end or self.agent_list[i].next_satellite_id == None:
                continue
            x, y = self.agent_list[i].curr_satellite_id, self.agent_list[i].next_satellite_id
            key = str(x) + "-" + str(y)
            band = self.satellite_list[x].idToBand[y]
            bandGet = self.agent_list[i].data_amount / map[key] * band
            transTime = self.agent_list[i].data_amount / bandGet
            self.agent_list[i].arrive_time = self.time + transTime / self.timeSlot
            self.agent_list[i].isTransmitting = True
            reward[i] = -transTime #+ self.alpha * self.agent_list[i].data_amount

        return reward

    def update_agent_state(self):
        for i in range(self.n_agents):
            if self.agent_list[i].curr_satellite_id != self.end and self.agent_list[i].arrive_time != None and self.agent_list[i].arrive_time <= self.time:
                if self.agent_list[i].next_satellite_id != None:
                    self.agent_list[i].curr_satellite_id = self.agent_list[i].next_satellite_id 
                self.agent_list[i].next_satellite_id = None
                self.agent_list[i].isTransmitting = False

    def update_satellite_state(self):
        if self.time % 5 == 0:
            for i in range(self.satellite_num):
                self.satellite_list[i].changeBandwiths()

    def step(self, actions):
        reward = self.take_action(actions)
        # reward = reward.sum()
        self.time += 1
        self.update_satellite_state()
        self.update_agent_state()

        obs_next = self.get_state()
        arrived = True
        for agent in self.agent_list:
            if agent.curr_satellite_id != self.end:
                arrived = False
                break
        # TODO: 添加数据压缩的负向奖励
        # print(reward)
        reward = reward.sum() if not arrived else reward.sum() + 100
        done = arrived or self.time >= self.episode_limit

        information = {}

        return reward, done, information

    def reset(self):
        self.time = 0
        self.agent_list = []
        for i in range(self.n_agents):
            data_amount = random.randint(50, 100)
            self.agent_list.append(DataStream(i, i, data_amount))

    def generateAdj(self):
        # 创建一个50x50的全零矩阵
        adj_matrix = np.zeros((50, 50), dtype=int)

        # 根据规则设置邻接关系
        for i in range(50):
            orbit = i // 10  # 计算卫星所在的轨道
            satellite_in_orbit = i % 10  # 计算卫星在轨道上的位置

            # 添加与同轨道的两颗卫星相连的边
            if satellite_in_orbit > 0:
                adj_matrix[i, i - 1] = 1
                adj_matrix[i - 1, i] = 1

            # 添加与相邻轨道的两颗卫星相连的边
            if orbit > 0:
                adj_matrix[i, i - 10] = 1
                adj_matrix[i - 10, i] = 1
            if orbit < 4:
                adj_matrix[i, i + 10] = 1
                adj_matrix[i + 10, i] = 1

        # 确保每一行刚好有四个1
        for i in range(50):
            row_sum = np.sum(adj_matrix[i])
            if row_sum < 4:
                indices = np.where(adj_matrix[i] == 0)[0]
                np.random.shuffle(indices)
                for j in range(4 - row_sum):
                    adj_matrix[i, indices[j]] = 1

        # 设置打印选项，打印全部内容
        np.set_printoptions(threshold=np.inf)

        # 打印整个邻接矩阵，不换行显示
        # for row in adj_matrix:
        #     #     print(row.sum())
        #     for elem in row:
        #         print(elem, end=' ')
        #     print("")

        ans = []
        m, n = adj_matrix.shape
        for i in range(m):
            neighbors = []
            for j in range(n):
                if adj_matrix[i][j] == 1.0:
                    neighbors.append(j)
            ans.append(neighbors)
        return ans
    
    def get_obs(self):
        """ Returns all agent observations in a list """
        states = []
        for i in range(self.n_agents):
            state = np.zeros(self.observation_space)
            state[0] = self.agent_list[i].data_amount
            state[1] = self.agent_list[i].curr_satellite_id
            state[2] = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor_bandwidths[0]
            state[3] = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor_bandwidths[1]
            state[4] = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor_bandwidths[2]
            state[5] = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor_bandwidths[3]
            state[6:-1] = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor
            state[-1] = self.end
            states.append(state)
        return states

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_state(self):
        # print(self.get_obs())
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
        return obs_concat
        
    def get_avail_actions(self):
        return [[1] * 7 for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return [1] * 7


    def get_state_size(self):
        return self.get_obs_size() * self.n_agents
    
    def get_obs_size(self):
        return self.observation_space
    
    def get_total_actions(self):
        return self.n_actions
    
    def get_stats(self):
        return None
    

if __name__ == '__main__':

    env = myEnv()
    env.reset()
    print(env.get_env_info())
    cur_time_step = 0
    while True:
        obs = env.get_obs()
        # print(obs)
        action = [random.randint(0, 6) for _ in range(5)]
        reward, done, _ = env.step(action)
        print(cur_time_step, action, reward)
        if done: 
            break
        cur_time_step += 1
        # print(cur_time_step, action, reward)
        

        


