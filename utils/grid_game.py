import numpy as np
from utils.agent import agent
import copy
import os
import json
import itertools

class gridForeging:
    def __init__(self, agent_position, gridmap, horizon, iter_num=2, beta = .5, 
            isparallel = True, optimal = False):
        self.agent_list = []
        self.gridMap = gridmap
        M, N = gridmap.shape
        self.Xlim, self.Ylim = M - 1, N - 1
        self.horizon = horizon
        self.beta = beta
        self.teamnum = len(agent_position)
        for i in range(self.teamnum):
            position = agent_position[i]
            self.agent_list.append(agent(position, i, self.horizon, self.Xlim, self.Ylim, 
                self.teamnum, beta=self.beta))

        self.iter_num = iter_num
        self.record = {}
        self.timeline = list(range(20))
        self.sol = {"policy": [], "utility": []}
        self.action_space = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
        self.action_space_team = [self.action_space] * self.teamnum
        self.all_possible_action_seq = list(itertools.product(*self.action_space_team))
        if optimal:
            self.optimal_simulate()
        else:
            if isparallel:
                self.iBR_simulate()
            else:
                self.iBR_simulate_seq()
    
    def iBR_simulate(self):
        # iBR in parallel way
        gridmap_copy = copy.deepcopy(self.gridMap)
        self.record['gridmap'] = gridmap_copy.tolist()
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
            record_iter["position"] = cur_positions
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
            cur_positions = []
            for i in range(self.teamnum):
                action = decision[i][0]
                position = self.agent_list[i].position
                new_position = self.dynamics(position, action)
                self.agent_list[i].position = new_position
                cur_positions.append(new_position)
            team_utility, self.gridMap = self.utility(cur_positions, self.gridMap)
            record_iter["gain"] = team_utility
            self.record[t] = record_iter

        dataPath = os.path.join(os.getcwd(), 'data', 'result')
        filename = os.path.join(dataPath, "ibr_iter_parallel_" + str (self.iter_num) + "_horizon_" + \
            str(self.horizon) + ".json")
        with open(filename, 'w') as outfiles:
            json.dump(self.record, outfiles, indent=4)

    def iBR_simulate_seq(self):
        # iBR in sequential way
        gridmap_copy = copy.deepcopy(self.gridMap)
        self.record['gridmap'] = gridmap_copy.tolist()
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
            record_iter["position"] = cur_positions
            record_iter["values"].append(self.u(cur_positions, self.gridMap, init_guess))
            record_iter["policyset"].append(init_guess)

            # 2. Interative Best Response
            for l in range(self.iter_num):
                # let each agent runs parallel of its B.R.
                br_policy = copy.deepcopy(policy_iter)
                for i in range(self.teamnum):
                    agent = self.agent_list[i]
                    br_policy[i] = agent.BestResponse(br_policy, self.gridMap, cur_positions)
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
            cur_positions = []
            for i in range(self.teamnum):
                action = decision[i][0]
                position = self.agent_list[i].position
                new_position = self.dynamics(position, action)
                self.agent_list[i].position = new_position
                cur_positions.append(new_position)
            team_utility, self.gridMap = self.utility(cur_positions, self.gridMap)
            record_iter["gain"] = team_utility
            self.record[t] = record_iter

        dataPath = os.path.join(os.getcwd(), 'data', 'result')
        filename = os.path.join(dataPath, "ibr_iter_seq_" + str (self.iter_num) + "_horizon_" + \
            str(self.horizon) + ".json")
        with open(filename, 'w') as outfiles:
            json.dump(self.record, outfiles, indent=4)

    def optimal_simulate(self):
        """ Exhaustive method to search for best result, DFS approach with pruning """                
        gridmap_copy = copy.deepcopy(self.gridMap)
        self.record['gridmap'] = gridmap_copy.tolist()
        cur_positions = self.getPositions()
        for t in self.timeline:
            # 1. search the best solution within the horizon
            
            record_iter = {}
            record_iter["position"] = cur_positions
            iter_policy = []
            for i in range(self.teamnum):
                iter_policy.append([])
            self.DFS(cur_positions, 0.0, iter_policy, self.gridMap, 0)
            # 2. DFS done, pick up the best soluion
            index = self.sol["utility"].index(max(self.sol["utility"]))
            decision = self.sol["policy"][index]
            record_iter["values"] = self.sol["utility"][index]
            record_iter["decision"] = decision
            
            # conduct final policy and change self.gridMap
            # make the decision happens
            cur_positions = []
            for i in range(self.teamnum):
                action = decision[i][0]
                position = self.agent_list[i].position
                new_position = self.dynamics(position, action)
                self.agent_list[i].position = new_position
                cur_positions.append(new_position)
            team_utility, self.gridMap = self.utility(cur_positions, self.gridMap)
            record_iter["gain"] = team_utility
            self.record[t] = record_iter
            # reset
            self.sol = {"policy": [], "utility": []}
            
        # save the data
        dataPath = os.path.join(os.getcwd(), 'data', 'result')
        filename = os.path.join(dataPath, "dfs_horizon_" + \
            str(self.horizon) + ".json")
        with open(filename, 'w') as outfiles:
            json.dump(self.record, outfiles, indent=4)

    def DFS(self, cur_positions, value, iter_policy, gridmap, t):
        for policy in self.all_possible_action_seq:
            new_policy = copy.deepcopy(iter_policy)
            new_position = []
            for i in range(self.teamnum):
                action = policy[i]
                pos = self.dynamics(cur_positions[i], action)
                if self.outside_map(pos):
                    break
                new_position.append(pos)
                new_policy[i].append(action)
            if len(new_position) == self.teamnum:
                # valid solution
                team_utility, new_gridmap = self.utility(new_position, copy.deepcopy(gridmap))
                new_value = value + sum(team_utility)
                if t + 1 == self.horizon:
                    self.sol["policy"].append(new_policy)
                    self.sol["utility"].append(new_value)
                else:
                    self.DFS(new_position, new_value, new_policy, new_gridmap, t+1)

    def outside_map(self, pos):
        """ check if single agent outside map"""
        if pos[0] < 0 or pos[0] > self.Xlim or pos[1] < 0 or pos[1] > self.Ylim:
            return True
        return False

    def getPositions(self):
        position_list = []
        for agent in self.agent_list:
            position_list.append(agent.position)
        return position_list
    
    def u(self, position_list, gridmap, action_list):
        gridmap_ = copy.deepcopy(gridmap)
        value = 0.0
        for t in range(self.horizon):
            new_position = []
            for i in range(self.teamnum):
                position = self.dynamics(position_list[i], action_list[i][t])
                new_position.append(position)
            position_list = new_position
            old_sum = gridmap_.sum()
            gain_list, gridmap_ = self.utility(position_list, gridmap_)
            new_sum = gridmap_.sum()
            if not (np.isclose((new_sum + sum(gain_list) - old_sum), 0.0)):
                raise RuntimeError("matrix calculation error!")
            value += sum(gain_list)
        return value

    def utility(self, agent_positions, gridmap):
        # agent_positions is a list contains tuple values
        position_record = {}
        for position in agent_positions:
            try:
                position_record[position] += 1
            except:
                position_record[position] = 1
        
        # for greedy_policy function
        # for key in position_record.keys():
        #     gain_per_agent = self.f(position_record[key], gridmap[key])
        #     gridmap[key] -= gain_per_agent * position_record[key]
        #     value += gain_per_agent * position_record[key]

        # calculate each agent's gain
        agent_gain = []
        previous_value = []
        for i in range(self.teamnum):
            pos = agent_positions[i]
            gain = self.f(position_record[pos], gridmap[pos])
            agent_gain.append(gain)
            previous_value.append(gridmap[pos])
        
        # then change the gridmap
        for i in range(self.teamnum):
            pos = agent_positions[i]
            gridmap[pos] -= agent_gain[i]
        
        return agent_gain, gridmap
    
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