import numpy as np
from utils.agent import agent
import copy

class gridForeging:
    def __init__(self, agent_position, gridmap, horizon, iter_num=2, beta = .5):
        self.agent_list = []
        self.gridMap = gridmap
        M, N = gridmap.shape
        self.horizon = horizon
        self.beta = beta
        self.teamnum = len(agent_position)
        for i in range(self.teamnum):
            position = agent_position[i]
            self.agent_list.append(agent(position, i, self.horizon, M - 1, N - 1, 
                self.teamnum, beta=self.beta))

        for i in range(len(agent_position)):
            print(str(self.agent_list[i]))
        self.iter_num = iter_num
        self.record = {}
        self.timeline = list(range(20))
        self.simulate()
        print(self.record)
    
    def simulate(self):
        # iBR in parallel way
        cur_positions = self.getPositions()
        for t in self.timeline:
            # 1. based on current gridmap, get init guesses for each agent
            init_guess = []
            isConverge = -1
            for agent in self.agent_list:
                greed_policy = agent.greedy_policy(self.gridMap)
                init_guess.append(greed_policy)
            
            policy_iter = copy.deepcopy(init_guess)
            record_iter = {"values": [], "policyset": [], "converge": isConverge}
            record_iter["values"].append(self.u(cur_positions, self.gridMap, init_guess))
            record_iter["policyset"].append(init_guess)

            # 2. Interative Best Response
            for l in range(self.iter_num):
                # let each agent runs parallel of its B.R.
                br_policy = []
                for agent in self.agent_list:
                    br_policy.append(agent.BestResponse(policy_iter, self.gridMap, cur_positions))
                # check if convergence, obtain a PNE
                if policy_iter == br_policy:
                    record_iter["converge"] = l
                    break
                policy_iter = br_policy
                record_iter["values"].append(self.u(cur_positions, self.gridMap, br_policy))
                record_iter["policyset"].append(br_policy)

            # conduct final policy and change self.gridMap
            if record_iter["converge"] > 0:
                decision = record_iter["policyset"][-1]
            else:
                # pick the best policy among all achienved policy, since this is a cooperative game
                index = record_iter["values"].index(max(record_iter["values"]))
                decision = record_iter["policyset"][index]
            record_iter["decision"] = decision

            # make the decision happens
            for i in range(self.teamnum):
                action = decision[i][0]
                position = self.agent_list[i].position
                new_position = self.dynamics(position, action)
                self.agent_list[i].position = new_position
                cur_positions.append(new_position)
            team_utility, self.gridMap = self.utility(cur_positions, self.gridMap)
            record_iter["gain"] = team_utility
            self.record[t] = record_iter
    
    def getPositions(self):
        position_list = []
        for agent in self.agent_list:
            position_list.append(agent.position)
        return position_list
    
    def u(self, position_list, gridmap, action_list):
        gridmap_ = gridmap
        value = 0.0
        for t in range(self.horizon):
            new_position = []
            for i in range(self.teamnum):
                position = self.dynamics(position_list[i], action_list[i][t])
                new_position.append(position)
            position_list = new_position

            gain, gridmap_ = self.utility(position_list, gridmap_)
            value += gain
        return value

    def utility(self, agent_positions, gridmap):
        # agent_positions is a list contains tuple values
        position_record = {}
        for position in agent_positions:
            try:
                position_record[position] += 1
            except:
                position_record[position] = 1
        value = 0.0
        
        # for greedy_policy function
        for key in position_record.keys():
            gain_per_agent = self.f(position_record[key], gridmap[key])
            gridmap[key] -= gain_per_agent * position_record[key]
            value += gain_per_agent * position_record[key]
        
        return value, gridmap
    
    def dynamics(self, position, action):
        
        x = position[0] + action[0]
        y = position[1] + action[1]
        return (x, y)

    def f(self, m, y):
        ## TODO
        return self.beta * y / m
        
    def checkConverge(self):
        # solves centralized planning in time horizon
        pass