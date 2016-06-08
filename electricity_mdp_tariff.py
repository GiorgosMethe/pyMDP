import numpy as np
import random
from domain import Variable, Domain
from learning import QLearning, SARSA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import retail_tariff as rt
import time

class ElectricityMDP(Domain):
    def __init__(self, k):
        Domain.__init__(self)
        self.charge_loss_factor = 0.9
        self.soc_external_losses = -0.05
        self.capacity_min = 0.1
        self.capacity_max = 0.3
        self.soc_min = 0.0
        self.soc_max = 1.0
        self.tau_min = 0.0
        self.tau_max = 1.0
        self.load_min = 0.0
        self.load_max = 1.0
        self.charge_min = 0.0
        self.charge_max = 0.5
        self.load_mean = 0.5
        self.load_std = 0.8

        self.initial_value = 0.0

        # state and action variables
        soc = Variable(self.soc_min, self.soc_max, 0.01) # state of charge
        capacity = Variable(self.capacity_min, self.capacity_max, 0.1) # consumption load
        a_tau = Variable(self.tau_min, self.tau_max, k) # consumption load
        charge = Variable(self.charge_min, self.capacity_max, 0.1) # consumption load

        self.tariff = rt.receive_tariff()

        self.setStateVariables({'soc':soc, 'capacity':capacity}) # define the state variables
        self.setActionVariables({'charge':charge, 'tau':a_tau}) # sets the action variables
        self.current_state = {'capacity':self.capacity_max, 'soc':self.soc_max} # initial state
        self.previous_state = None
        self.new_capacity = self.capacity_max

    def step(self, action):
        reward = 0.0
        new_state = {'capacity':self.new_capacity, 'soc':0.0}
        self.previous_state = self.current_state

        tau = action['tau'] # change tariff
        tau = 0.0 # change tariff

        new_state['soc'] = max(self.soc_min, (self.current_state['soc'] + (self.current_state['soc'] * self.soc_external_losses)))
        observed_load = rt.rvs()

        if random.random() < 0.01:
            if random.random() > 0.5:
                self.new_capacity = self.current_state['capacity'] + 0.1
            else:
                self.new_capacity = self.current_state['capacity'] - 0.1
            self.new_capacity = max(self.capacity_min, min(self.capacity_max, new_state['capacity']))
        # new soc state depending on the capacity of the storage
        new_state['soc'] = min(self.soc_max, self.current_state['soc'] * (self.current_state['capacity'] / new_state['capacity']))

        current_soc = new_state['soc'] * new_state['capacity']
        current_soc = current_soc + (self.charge_loss_factor * action['charge'])
        soc_imbalance = current_soc - new_state['capacity']
        current_soc = min(new_state['capacity'], current_soc)
        new_state['soc'] = current_soc / new_state['capacity']

        imbalance = observed_load - rt.mean()
        current_soc = new_state['soc'] * new_state['capacity']
        rest_storage = new_state['capacity'] - current_soc
        if imbalance < 0:
            if rest_storage > abs(imbalance):
                current_soc = current_soc + abs(imbalance)
                imbalance = 0.0
            else:
                imbalance = imbalance + abs(new_state['capacity'] - current_soc)
                current_soc = current_soc + abs(new_state['capacity'] - current_soc)
        else:
            if current_soc >= imbalance:
                imbalance = 0.0
                current_soc = current_soc - imbalance
            else:
                imbalance = imbalance - current_soc
                current_soc = 0.0
        new_state['soc'] = current_soc / new_state['capacity']

        # final imbalance
        reward = - (rt.mean() * self.tariff[tau])
        reward = reward - (action['charge'] * 0.12)
        reward = reward - ((1.0 - tau) * abs(imbalance) * 0.52) - (abs(soc_imbalance) * 0.52)
        ## Check for next state
        if new_state['capacity'] < self.capacity_min or new_state['capacity'] > self.capacity_max: raise AssertionError()
        if new_state['soc'] < self.soc_min or new_state['soc'] > self.soc_max: raise AssertionError()
        self.current_state = new_state
        return reward

    def Simulate(mdp, initial_state, learner, epsilon, steps, nsteps, data):
        pass
