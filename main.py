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
    
    # matrix1 = [[.143, .808, .754, .202, .564, .485],
    #             [.456, .769, .136, .354, .791, .750],
    #             [.481, .841, .419, .451, .917, .347],
    #             [.479, .331, .337, .778, .768, .758],
    #             [.933, .041, .097, .170, .630, .593],
    #             [.003, .861, .115, .054, .075, .144]]
    # gridmap = np.matrix(matrix1) 
    
    M = 6
    rng = np.random.default_rng(seed=0) #generate a random seed
    gridmap = rng.random((M,M))
    gridForeging(agent_position, copy.deepcopy(gridmap), 2, 
                        iter_num=4, isparallel=True, optimal=False, trial=0)

    # statistical result
    # for i in range(1, 100):
    #     print("i = %s"%i)
    #     rng = np.random.default_rng(seed=i) #generate a random seed
    #     gridmap = rng.random((M,M))
    #     for horizon in range(3, 5):
    #         # PNE approach
    #         for iter_ in range(2, 5):
    #             for parallel in [True, False]:
    #                 gridForeging(agent_position, copy.deepcopy(gridmap), horizon, 
    #                     iter_num=iter_, isparallel=parallel, optimal=False, trial=i)
    #         # DFS approach
    #         if horizon < 4:
    #             print("I'm here")
    #             gridForeging(agent_position, copy.deepcopy(gridmap), horizon, 
    #                             isparallel=False, optimal=True, trial=i)
    #             print("I'm done")

    


if __name__=="__main__":
    start()
