import numpy as np
import itertools
import copy

class agent:
    def __init__(self, position, ID, horizon, Xlim, Ylim, teamnum, beta = 0.5):
        self.position = (position[0], position[1])
        self.action_space = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
        self.id = ID
        self.horizon = horizon        
        self.horizon_action_space = [self.action_space] * self.horizon
        self.all_possible_action_seq = list(itertools.product(*self.horizon_action_space))
        self.beta = beta
        self.alpha = 0.5
        self.Xlim, self.Ylim = Xlim, Ylim
        self.teamnum = teamnum
        
    
    def dynamics(self, position, action):
        
        x = position[0] + action[0]
        y = position[1] + action[1]
        return (x, y)
    
    def greedy_policy(self, gridmap):
        outcomes = []
        for i in range(len(self.all_possible_action_seq)):
            gridmap_ = copy.deepcopy(gridmap)
            action_seq = self.all_possible_action_seq[i]
            value = 0.0
            position = self.position
            for action in action_seq:
                position = self.dynamics(position, action)
                if self.outside_map([position]):
                    value = 0
                    break
                gain, gridmap_ = self.utility([position], gridmap_)
                value += gain
            outcomes.append(value)
        # find the maximal value in outcomes then output
        max_index = outcomes.index(max(outcomes))
        return self.all_possible_action_seq[max_index]
    
    def BestResponse(self, policy_iter, gridmap, position_list):
        # precalculate all other agents positions in future
        future_pos = {}
        for i in range(self.teamnum):
            if i != self.id:
                position = position_list[i]
                policy = policy_iter[i]
                future_pos[i] = []
                for action in policy:
                    position = self.dynamics(position, action)
                    future_pos[i].append(position)

        # then seach for the best response for ego agent
        outcomes = []
        for i in range(len(self.all_possible_action_seq)):
            gridmap_ = copy.deepcopy(gridmap)
            action_seq = self.all_possible_action_seq[i]
            value = 0.0
            position = self.position
            for t in range(self.horizon):
                action = action_seq[t]
                position = self.dynamics(position, action)
                if self.outside_map([position]):
                    value = 0
                    break
                position_list = []
                for ids in range(self.teamnum):
                    if ids == self.id:
                        position_list.append(position)
                    else:
                        position_list.append(future_pos[ids][t])

                gain, gridmap_ = self.utility(position_list, gridmap_)
                value += gain
            outcomes.append(value)
        # find the maximal value in outcomes then output
        max_index = outcomes.index(max(outcomes))
        return self.all_possible_action_seq[max_index]
    
    def outside_map(self, position_list):

        for pos in position_list:
            if pos[0] < 0 or pos[0] > self.Xlim or pos[1] < 0 or pos[1] > self.Ylim:
                return True
        return False

    def utility(self, agent_positions, gridmap):
        # agent_positions is a list contains tuple values
        position_record = {}
        for position in agent_positions:
            try:
                position_record[position] += 1
            except:
                position_record[position] = 1
        value = 0.0
        # extract ego agent info gain, and make weights on it
        if len(agent_positions) > 1:
            ego_pos = agent_positions[self.id]
            for key in position_record.keys():
                gain_per_agent = self.f(position_record[key], gridmap[key])
                gridmap[key] -= gain_per_agent * position_record[key] * self.alpha
                if key == ego_pos:
                    value += gain_per_agent
                    value += (position_record[key] - 1) * self.alpha * gain_per_agent
                else:
                    value += gain_per_agent * position_record[key] * self.alpha
                
        else:
            # for greedy_policy function
            for key in position_record.keys():
                
                gain_per_agent = self.f(position_record[key], gridmap[key])
                gridmap[key] -= gain_per_agent * position_record[key]
                value += gain_per_agent * position_record[key]
        
        return value, gridmap
    
    def f(self, m, y):
        ## TODO
        return self.beta * y / m

        
    
    def __str__(self):
        return "I\'m agent %s, I'm in postion %s." % (self.id, self.position)