import numpy as np
from utils.grid_game import gridForeging
import os, json
import copy

def start():
    
    path = os.getcwd()
    filename = os.path.join(path, 'data', 'world', 'gridmap.json')
    
    with open(filename) as json_file:
        data = json.load(json_file)

    agent_position = data["agent_position"]
    
    horizon = data["horizon"]
    
    matrix1 = [[.143, .808, .754, .202, .564, .485],
                [.456, .769, .136, .354, .791, .750],
                [.481, .841, .419, .451, .917, .347],
                [.479, .331, .337, .778, .768, .758],
                [.933, .041, .097, .170, .630, .593],
                [.003, .861, .115, .054, .075, .144]]
    gridmap = np.matrix(matrix1) 

    gridForeging(agent_position, copy.deepcopy(gridmap), 3, 
                    isparallel=False, optimal=True)
    
    # for horizon in range(3, 6):
    #     for iter_ in range(2, 5):
    #         for parallel in [True, False]:
    #             gridForeging(agent_position, copy.deepcopy(gridmap), horizon, 
    #                 iter_num=iter_, isparallel=parallel)

    


if __name__=="__main__":
    start()
