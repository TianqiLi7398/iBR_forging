import numpy as np
import matplotlib.pyplot as plt
import json
import os
import matplotlib.pyplot as plt

def compare_horizon_iter():
    path = os.getcwd()
    
    T = 20
    timeline = np.linspace(1, T, T)
    dataset = {"seq": {}, "parallel": {}}
    for key in dataset.keys():
        for horizon in [3, 4]:
            dataset[key][horizon] = {}
            for iter_ in range(2, 5):
                dataset[key][horizon][iter_] = [0.0] * T
        
                # record files
                for i in range(100):
                    name = 'ibr_iter_' + key + '_' + str(iter_) + '_horizon_' + str(horizon) + '_' + str(i)
                    filename = os.path.join(path, 'data', 'result', name + '.json')
                    with open(filename) as json_file:
                        data = json.load(json_file)
                    for t in range(T):
                        dataset[key][horizon][iter_][t] += sum(data[str(t)]["gain"])
                    

                # cpf
                accumulated_gain = np.zeros(T)
                for t in range(T):
                    if t > 0:
                        accumulated_gain[t] = dataset[key][horizon][iter_][t] + accumulated_gain[t - 1]
                    else:
                        accumulated_gain[t] = dataset[key][horizon][iter_][t]
                accumulated_gain /= 100
                plt.plot(timeline, accumulated_gain, label = key + ' H = ' + str(horizon) + ', L = ' + str(iter_))
    
        picname = os.path.join(path, 'pics',key +'_horizon_iter.png' )
        plt.xticks(timeline)  
        plt.xlabel("Time Step")
        plt.ylabel("Average Accumulated System Gain")
        plt.legend()
        plt.grid()
        plt.savefig(picname)
        plt.close()

def ibr_dfs():
    horizon = 3
    iter_ = 4
    key = 'seq'
    path = os.getcwd()
    T = 20
    timeline = np.linspace(1, T, T)
    trial_num = 100
    color_bar = ['r', 'b', 'g', 'yellow']
    # IBR data
    for j in range(2):
        key =  ['seq', 'parallel'][j]
        
        ibr = []
        for i in range(trial_num):
            trial_record = []
            name = 'ibr_iter_' + key + '_' + str(iter_) + '_horizon_' + str(horizon) + '_' + str(i)
            filename = os.path.join(path, 'data', 'result', name + '.json')
            with open(filename) as json_file:
                data = json.load(json_file)
            for t in range(T):
                if t > 0:
                    trial_record.append(trial_record[t-1] + sum(data[str(t)]["gain"]))
                else:
                    trial_record.append(sum(data[str(t)]["gain"]))
            ibr.append(trial_record)

        mean, std = np.mean(np.array(ibr), axis=0), np.std(np.array(ibr), axis=0)
        
        conf_inter = 1.96 * std / np.sqrt(trial_num)
        
        plt.plot(timeline, mean, color=color_bar[j], label = key + ' H = ' + str(horizon) + ', L = ' + str(iter_))
        plt.fill_between(timeline, mean - conf_inter, mean+ conf_inter, 
                    color=color_bar[j], alpha=.1)

    # DFS data
    dfs = []
    for i in range(trial_num):
        trial_record = []
        name = 'dfs_horizon_' + str(horizon) + '_' + str(i)
        filename = os.path.join(path, 'data', 'result', name + '.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        for t in range(T):
            if t > 0:
                trial_record.append(trial_record[t-1] + sum(data[str(t)]["gain"]))
            else:
                trial_record.append(sum(data[str(t)]["gain"]))
        dfs.append(trial_record)

    # cpf
    mean, std = np.mean(np.array(dfs), axis=0), np.std(np.array(dfs), axis=0)
        
    conf_inter = 1.96 * std / np.sqrt(trial_num)
    
    plt.plot(timeline, mean, color=color_bar[2], label = 'DFS H = ' + str(horizon))
    plt.fill_between(timeline, mean - conf_inter, mean+ conf_inter, 
                color=color_bar[2], alpha=.1)
    
    picname = os.path.join(path, 'pics', 'ibrVSdfs.png' )
    plt.xlabel("Time step")
    plt.ylabel("Average Accumulated System Gain")
    plt.xticks(timeline)  
    plt.legend()
    plt.grid()
    plt.savefig(picname)
    plt.close()

def pne_converge():
    path = os.getcwd()
    
    T = 20
    timeline = np.linspace(1, T, T)
    
    for key in ['seq', 'parallel']:
        print(key)
        for horizon in [3, 4]:
            
            for iter_ in range(2, 5):
                converge_account = 0.0
                runtime = 0
                # record files
                for i in range(100):
                    name = 'ibr_iter_' + key + '_' + str(iter_) + '_horizon_' + str(horizon) + '_' + str(i)
                    filename = os.path.join(path, 'data', 'result', name + '.json')
                    with open(filename) as json_file:
                        data = json.load(json_file)
                    for t in range(T):
                        if data[str(t)]["converge"] > -1:
                            converge_account += 1
                    runtime += data['t']
                # cpf
                runtime *= 0.01
                converge_rate = converge_account / (T * 100)
                print("PNE finding rate for horizon = %s, iter_num = %s is %s" % (horizon, iter_, converge_rate))
                print("average runtime = %s" % runtime)

def compare_agent():
    path = os.getcwd()
    agent_num = 3
    T = 20
    timeline = np.linspace(1, T, T)
    dataset = {"seq": {}, "parallel": {}}
    horizon = 3
    iter_ = 4
    trial_num = 100
    fig, axs = plt.subplots(agent_num, 1, figsize=(12, 8), sharex=True)
    color_bar = ['r', 'b', 'g', 'yellow']
    for j in range(2):
        key =  ['seq', 'parallel'][j]
       
        individual_gain = []
        for i in range(agent_num):
            individual_gain.append([])
        # record files
        for i in range(trial_num):
            name = 'ibr_iter_' + key + '_' + str(iter_) + '_horizon_' + str(horizon) + '_' + str(i)
            filename = os.path.join(path, 'data', 'result', name + '.json')
            with open(filename) as json_file:
                data = json.load(json_file)
            record = [[], [], []]
            for t in range(T):
                average_gain = sum(data[str(t)]["gain"])/3
                for ag in range(agent_num):
                    diverage = average_gain - data[str(t)]["gain"][ag]
                    record[ag].append(diverage)
            for ag in range(agent_num):
                individual_gain[ag].append(record[ag])
        # cov_individual = np.sqrt(average_individual)
        
        for ag in range(agent_num):
            mean, std = np.mean(np.array(individual_gain[ag]), axis=0), np.std(np.array(individual_gain[ag]), axis=0)
        
            conf_inter = 1.96 * std / np.sqrt(trial_num)
            axs[ag].plot(timeline, mean, color=color_bar[j], label = key + ' H = ' + str(horizon) + ', L = ' + str(iter_))
            axs[ag].fill_between(timeline, mean - conf_inter, mean + conf_inter, 
                        color=color_bar[j], alpha=.1)
            # axs[ag].plot(timeline, cov_individual[ag, :], label = key + ' horizon = ' + str(horizon) + ' iter_num = ' + str(iter_))
            # print(key + " horizon = %s, iter_num = %s contains balance of %s" %(horizon, iter_, np.sum(cov_individual, axis=1)))
            print(key + " horizon = %s, iter_num = %s contains balance of %s" %(horizon, iter_, np.sum(mean)))

    # dfs
    individual_gain = np.zeros((3, T))
    runtime = 0
    individual_gain = []
    for i in range(agent_num):
        individual_gain.append([])
    for i in range(trial_num):
        name = 'dfs_horizon_' + str(horizon) + '_' + str(i)
        filename = os.path.join(path, 'data', 'result', name + '.json')
        with open(filename) as json_file:
            data = json.load(json_file)
        record = [[], [], []]
        for t in range(T):
            average_gain = sum(data[str(t)]["gain"])/3
            for ag in range(agent_num):
                diverage = average_gain - data[str(t)]["gain"][ag]
                record[ag].append(diverage)
        for ag in range(agent_num):
            individual_gain[ag].append(record[ag])
        runtime += data['t']
    # calculate the covariance thing
    for ag in range(agent_num):
        mean, std = np.mean(np.array(individual_gain[ag]), axis=0), np.std(np.array(individual_gain[ag]), axis=0)
    
        conf_inter = 1.96 * std / np.sqrt(trial_num)
        axs[ag].plot(timeline, mean, color=color_bar[2], label = 'DFS H = ' + str(horizon))
        axs[ag].fill_between(timeline, mean - conf_inter, mean + conf_inter, 
                    color=color_bar[2], alpha=.1)
        print("DFS horizon = %s contains balance of %s" %(horizon, np.sum(mean)))
    print("DFS runtime = %s" % (runtime * 0.01))

    for ag in range(agent_num):
        axs[ag].set_title("Agent %s" %(ag + 1), fontweight='bold')
        axs[ag].legend()
        axs[ag].grid()
        axs[ag].set_xlabel("Time Step")
        axs[ag].set_ylabel("Average Divergance")
        axs[ag].set_xticks(timeline)
    picname = os.path.join(path, 'pics','agent_compare.png' )
    # plt.xticks(timeline)  
    # plt.xlabel("Time step")
    # plt.ylabel("Average Accumulated System Gain")
    # plt.legend()
    # plt.grid()
    plt.savefig(picname)
    plt.close()




def analysis():
    # compare the horizon length to total utility gain
    # compare the iteration length
    compare_horizon_iter()

    # compare the iBR v.s. DFS
    ibr_dfs()

    # check statistical result of agents utility
    compare_agent()
    # convergence rate
    pne_converge()

if __name__=="__main__":
    analysis()
