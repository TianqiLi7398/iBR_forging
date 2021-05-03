import numpy as np
import copy
import os
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.text import Text

def start(iter_num, horizon, parallel, iter):
    
    dataPath = os.path.join(os.getcwd(), 'data', 'result')
    if parallel:
        filename = os.path.join(dataPath, "ibr_iter_parallel_" + str (iter_num) + "_horizon_" + \
            str(horizon) + "_" + str(iter) + ".json")
    else:
        filename = os.path.join(dataPath, "ibr_iter_seq_" + str (iter_num) + "_horizon_" + \
            str(horizon) + ".json")
    with open(filename) as json_file:
        record = json.load(json_file)

    time = len(record.keys()) - 1
    gridmap = np.matrix(record["gridmap"])

    
    global ax, fig
    fig = plt.figure()
    #fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal')
    trj1, = ax.plot([], [], 'ko', ms=2)

    def init():
        trj1.set_data([], [])
        return trj1,

    def animate(t):

        for obj in ax.findobj(match = FancyArrowPatch):
            obj.remove()

        for obj in ax.findobj(match = Circle):
            obj.remove()
        
        for obj in ax.findobj(match = Text):
            
            obj.set_visible(False)

        ax.matshow(gridmap, cmap=plt.cm.Blues)
        t = str(t)
        position = record[t]["position"]
        decision = record[t]["decision"]
        ax.text(2.5, -1.5, 'time step ' + t)

        for i in range(gridmap.shape[1]):
            for j in range(gridmap.shape[0]):
                c = round(gridmap[j,i], 3)
                ax.text(i, j, str(c), va='center', ha='center')

        for i in range(len(position)):
            pos = position[i]
            action = decision[i][0]
            circle = Circle((pos[1], pos[0]), 0.2, color = 'r', fc=None)
            ax.add_artist(circle)
            new_pos = [pos[0] + action[0], pos[1] + action[1]]
            e = FancyArrowPatch((pos[1], pos[0]), (new_pos[1], new_pos[0]),
							arrowstyle='<-',
							linewidth=2,
							color='k')
            ax.add_artist(e)
            # dynamics of the map
            gridmap[new_pos[0], new_pos[1]] -= record[t]["gain"][i]
        
        return trj1,

    
    ani = animation.FuncAnimation(fig, animate, frames=time-1,
                                interval=10, blit=True, init_func=init, repeat = False)
    path = os.getcwd()

    videopath = os.path.join(path, 'video')
    if parallel:
        filename = os.path.join(videopath, "ibr_iter_parallel_" + str (iter_num) + "_horizon_" + \
            str(horizon) + "_" + str(iter) + '.mp4')
    else:
        filename = os.path.join(videopath, "ibr_iter_seq_" + str (iter_num) + "_horizon_" + \
            str(horizon) + '.mp4')
    ani.save(filename, fps=2)
    plt.close()
    


if __name__=="__main__":
    iter_num, horizon = 4, 2
    parallel = True
    start(iter_num, horizon, parallel, 0)
    # for horizon in range(3, 6):
    #     for iter_ in range(2, 5):
    #         for parallel in [True, False]:
    #             start(iter_, horizon, parallel)
