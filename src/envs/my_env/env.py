import numpy as np
import random
import gym
from gym import spaces

import sys
import os
from types import SimpleNamespace as SN
import torch as th

import collections
# 添加父目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

from multiagentenv import MultiAgentEnv
from satellite import Satellite
from dataStream import DataStream




class myEnv(MultiAgentEnv):
    # 50个卫星 5个源 1个end
    def __init__(
        self,
        episode_limit=100,
        n_agents=5,
        satellite_num=50,
        **kwargs,
        ):
        super(myEnv, self).__init__()
        self.time = 0 # 时隙的轮次
        self.timeSlot = 1 # 每个时隙的大小
        self.episode_limit = episode_limit # 最大时隙
        self.n_agents = n_agents
        self.n_actions = 8 # 1 + 4 + 3
        self.satellite_num = satellite_num
        self.adj = self.readAdj()  # 邻接矩阵
        self.satellite_list = []
        self.agent_list = []
        self.compression_ratio_list = [0, 0.05, 0.1] # 图像压缩率的列表
        self.alpha = 0.5  # 传输时间和数据量之间的权重系数
        self.observation_space = 1 + 1 + 4 + 4 + 1 + self.n_agents # 状态维度，数据量-1 + 所在卫星ID-1 + 四个邻居卫星的带宽-4 + 四个邻居卫星的ID-4 + 目的地-1 + 所有智能体的位置-5
        # self.action_space = 7
        # self.action_space = [spaces.Discrete([7]) for _ in range(self.n_agents)]
        # self.observation_space = spaces
        self.end = 20 #random.randint(self.n_agents, self.satellite_num - 1)  # 前n_agents个卫星是EO卫星，从后面的id中随机生成一个终点
        self.visited = [np.zeros(satellite_num) for _ in range(n_agents)]
        self.arrived = []
        self.time_cost = 0
        self.data_residule = 0
        self.max_time = 100

        for i in range(self.satellite_num):  # 生成所有的卫星对象
            neighbor_ids = self.adj[i]
            neighbor_bandwidths = []
            for j in range(4):
                neighbor_bandwidths.append(10)
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
            data_amount = 100 #random.randint(50, 100)
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
            
            if i_agent_action < 5:
                target = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor[i_agent_action-1]
                self.agent_list[i].next_satellite_id = target  # 修改下一跳目的
                # print(f"agent {i} transfer to {target}, end is {self.end}, time_step is {self.time}")
            else:
                self.agent_list[i].data_amount = max(self.agent_list[i].min_data_amount, self.agent_list[i].data_amount*(1 - self.compression_ratio_list[i_agent_action - 5]))  # 修改数据量
                # reward[i] -= self.compression_ratio_list[i_agent_action-4]*100


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
            reward[i] -= transTime #+ self.alpha * self.agent_list[i].data_amount

        return reward

    def update_agent_state(self):
        for i in range(self.n_agents):
            self.visited[i][self.agent_list[i].curr_satellite_id] = 1
            if self.agent_list[i].curr_satellite_id != self.end and self.agent_list[i].arrive_time != None and self.agent_list[i].arrive_time <= self.time:
                if self.agent_list[i].next_satellite_id != None:
                    self.agent_list[i].curr_satellite_id = self.agent_list[i].next_satellite_id 
                self.agent_list[i].next_satellite_id = None
                self.agent_list[i].isTransmitting = False
            self.visited[i][self.agent_list[i].curr_satellite_id] = 1

    def update_satellite_state(self):
        if self.time % 5 == 0:
            for i in range(self.satellite_num):
                self.satellite_list[i].changeBandwiths()

    def step(self, actions):
        reward = self.take_action(actions)
        reward = reward.sum()
        reward = reward / 10
        self.time += 1
        self.update_satellite_state()
        self.update_agent_state()

        obs_next = self.get_state()
        arrived = True
        for agent in self.agent_list:
            if agent.curr_satellite_id != self.end:
                arrived = False
            elif not agent.id in self.arrived:
                self.arrived.append(agent.id)
                reward += 50
                self.time_cost += self.time
                self.data_residule += agent.data_amount
                # print(agent.id)
        if arrived:
            self.max_time = self.time
        # TODO: 添加数据压缩的负向奖励
        done = arrived or self.time >= self.episode_limit
        penalty = 0
        if done:
            for agent in self.agent_list:
                penalty += agent.compute_data_loss_penalty()
            # print([agent.data_amount/agent.original_data_amount for agent in self.agent_list])
            # print(penalty)
            # vis = []
            # for i in range(self.satellite_num):
            #     flag = True
            #     for j in range(self.n_agents):
            #         if self.visited[j][i] == 0:
            #             flag = False
            #             break
            #     if flag:
            #         vis.append(i)
            # print(arrived, vis, self.arrived)
        penalty_regularized = penalty # 归一化 [0, 1)
        reward = reward + penalty_regularized
        # if arrived:
        #     reward += 10000
            # print([agent.curr_satellite_id for agent in self.agent_list])
        
        # reward = reward.sum() if not arrived else reward.sum() + 1000 + penalty*0.1
                
        information = {}

        return reward, done, information

    def reset(self):
        self.time = 0
        self.agent_list = []
        for i in range(self.n_agents):
            data_amount = random.randint(50, 100)
            self.agent_list.append(DataStream(i, i, data_amount))
        self.visited = [np.zeros(self.satellite_num) for _ in range(len(self.agent_list))]
        self.arrived = []
        self.time_cost = 0
        self.data_residule = 0
        self.max_time = 100

    def readAdj(self):
        data = []
        with open('/home/ruanzhonghai/work/Satellite/src/envs/my_env/adj_matrix.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                numbers = line.strip().split()
                int_numbers = [int(num) for num in numbers]
                data.append(int_numbers)
                
        return data
    
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
            state[6:10] = self.satellite_list[self.agent_list[i].curr_satellite_id].neighbor
            state[10] = self.end
            state[11:] = np.array([agent.curr_satellite_id for agent in self.agent_list])
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
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        avail_actions = [1] * self.n_actions
        if self.agent_list[agent_id].isTransmitting == True:
            avail_actions = [0] * self.n_actions
            avail_actions[0] = 1
        return avail_actions


    def get_state_size(self):
        return self.get_obs_size() * self.n_agents
    
    def get_obs_size(self):
        return self.observation_space
    
    def get_total_actions(self):
        return self.n_actions
    
    def get_stats(self):
        # print('get stats')
        stats = {
            'arrived': len(self.arrived)==5,
            'data_residule': self.data_residule,
            'time_cost': self.time_cost,
            'max_time': self.max_time,
        }
        for i in range(self.n_agents):
            if not i in self.arrived:
                stats["data_residule"] += self.agent_list[i].data_amount
                stats["time_cost"] += 50
        # if stats['arrived'] == True:
        #     stats['max_time'] = 
        # print(len(self.arrived))
        return stats


def test(args, logger=None):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # Setup multiagent controller here
    model_path = "/home/ruanzhonghai/work/Satellite/results/models/qmix_env=8_adam_td_lambda__2024-05-12_21-44-30/4013145"
    # model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))
    # logger.console_logger.info("Loading model from {}".format(model_path))
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
    # learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    mac.load_models(model_path)
    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    
    runner.run(test_mode=True)
    
def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        parent_dir = os.path.join(current_dir, os.pardir, os.pardir)
        with open(os.path.join(parent_dir, "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

if __name__ == '__main__':

    env = myEnv()
    env.reset()
    print(env.get_env_info())
    cur_time_step = 0
    ep = 1000
    res = []
    datas = []
    times = []
    max_times = []
    for i in range(ep):
        r = []
        env.reset()
        cur_time_step = 0
        while True:
            obs = env.get_obs()
            # print(obs)
            # binds = [o[2:6] for o in obs]
            # action = np.argmax(binds, axis=-1)
            
            # print(obs)
            action = [random.randint(0, 6) for _ in range(5)]
            # print(action)
            reward, done, _ = env.step(action)
            # print(cur_time_step, action, reward)
            r.append(reward)
            if done: 
                break
            cur_time_step += 1
       
        stats = env.get_stats()
        max_times.append(stats['max_time'])
        datas.append(stats['data_residule'])
        # times.append(stats['tim'])
        res.append(sum(r))
    print(sum(res)/ep)
    print(sum(max_times)/ep)
    print(sum(datas)/ep)
    
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # # 添加 parent_dir 到 sys.path
    # # os.path.join 用于构建路径，os.path.pardir 用于引用上级目录
    # parent_dir = os.path.join(current_dir, os.pardir, os.pardir)
    # sys.path.append(parent_dir)
    # print(parent_dir)
    # from runners import REGISTRY as r_REGISTRY
    # from learners import REGISTRY as le_REGISTRY
    # from controllers import REGISTRY as mac_REGISTRY
    # from components.episode_buffer import ReplayBuffer
    # from components.transforms import OneHot
    # from copy import deepcopy
    # import yaml

    # with open(os.path.join(parent_dir, "config", "default.yaml"), "r") as f:
    #     try:
    #         config_dict = yaml.load(f)
    #     except yaml.YAMLError as exc:
    #         assert False, "default.yaml error: {}".format(exc)
    # params = deepcopy(sys.argv)
    # env_config = _get_config(params, "--env-config", "envs")
    # alg_config = _get_config(params, "--config", "algs")
    # config_dict = recursive_dict_update(config_dict, env_config)
    # config_dict = recursive_dict_update(config_dict, alg_config)

    # args = SN(**config_dict)
    # test(args)

        

        


