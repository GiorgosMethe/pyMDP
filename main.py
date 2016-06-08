import numpy as np
import random
from domain import Variable, Domain
from learning import QLearning
from electricity_mdp_tariff import ElectricityMDP
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pickle
import time
import sys
import threading
import mutex
import copy

mutex = mutex.mutex()

def simulate(steps, nsteps, mdp, learner, data):
    sum_reward = 0.0
    avg_reward_array = []
    epsilon = 1.0
    epsilon_cof = 0.1
    for i in range(1, steps+1):
        action = learner.epsilonGreedy(mdp.current_state, 0.01)
        reward = mdp.step(action)
        learner.update(mdp.previous_state, action, mdp.current_state, reward)
        if i % (steps/nsteps) == 0:
            epsilon = epsilon * epsilon_cof
            avg_reward = (sum_reward + reward) / (steps/nsteps)
            print i / (steps/nsteps), '/', nsteps, ', iterations:', i, ', average reward:', avg_reward
            avg_reward_array.append(avg_reward)
            sum_reward = 0.0
        else:
            sum_reward = sum_reward + reward
    def append_thread_data(_data):
        data.append(_data)
        mutex.unlock()
    mutex.lock(append_thread_data, avg_reward_array)

def main():
    data = []
    thread_list = []
    repeats = 1
    steps = 1000000
    nsteps = 10
    max_num_threads = 8
    mdp = ElectricityMDP(0.1)
    learner = QLearning(mdp, 0.01, 0.3)
    if 0:
        while repeats > 0:
            if threading.active_count() - 1 < max_num_threads:
                repeats = repeats - 1 # remaining repeats
                p = threading.Thread(target=simulate, args=(steps, nsteps, copy.deepcopy(mdp), copy.deepcopy(learner), data))
                thread_list.append(p)
                p.start()
        for thread in thread_list:
            thread.join()
    else:
        simulate(steps, nsteps, copy.deepcopy(mdp), copy.deepcopy(learner), data)


    pkl_file = open( "data3.p", "wb")
    pickle.dump(data, pkl_file)
    pkl_file.close()

    sns.tsplot(time=range(1, steps+1, (steps/nsteps)),data=np.asarray(data), condition="Q-Learning", err_style="ci_bars", color="g")
    plt.xlabel('Iterations')
    plt.ylabel('Utility')
    plt.show()

if __name__ == "__main__":
    main()
