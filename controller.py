import numpy as np
import random
from domain import Variable, Domain
from learning import QLearning
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys

class ControllerMDP(Domain):
    def __init__(self):
        Domain.__init__(self)
        self.charge_loss_factor = 0.95
        self.soc_external_losses = -0.03
        self.initial_value = 10.0
        self.max_soc = 0.5
        self.min_soc = 0.0
        self.max_load = 1.0
        self.min_load = 0.0
        self.max_proc = 2.0
        self.min_proc = 0.0
        self.max_tariff, self.base_price, self.pr_coeff, self.pr_im_coeff = 9, 0.22, 0.02, 0.4
        self.imbalance = 0.0
        self.load_mean = 0.0
        self.load_std = 0.1

    def transition(self, state, action):
        """
        Inputs:
        Gets two dicts (state, action) and returns the new state of the system
        state: {'load': 0.4, 'soc': 1.0, 'tod': 2.0}
        action:  {'charge': -0.1}
        AND
        Returns: new_state: {'load': 0.4, 'soc': 1.0, 'tod': 2.0}
        """
        # Load at the next timestep
        new_load = state['load'] + np.random.normal(self.load_mean, self.load_std)
        new_load = max(self.min_load, min(self.max_load, new_load)) # keep it between 0.0 ~ 1.0

        # that's what we bought for the next hour-interval
        procurement = action['procure']
        excess_shortage = procurement - new_load # the difference will be stored or withdrawn from the storage

        # self.imbalance = 0.0
        # if excess_shortage < 0.0: # means the stored energy cannot make it up for the imbalance beween procured energy and actual consumption
        #     if state['soc'] < abs(excess_shortage):
        #         self.imbalance = abs(excess_shortage) - state['soc']
        self.imbalance = abs(excess_shortage)

        # State of charge at the next timestep
        new_soc = state['soc'] + (self.charge_loss_factor * excess_shortage) + (state['soc'] * self.soc_external_losses)
        new_soc = max(self.min_soc, min(self.max_soc, new_soc)) # keep it between 0.0 ~ 1.0

        # next tariff with 0 we stay with the current
        new_tariff = state['tariff']
        if action['tariff'] != 0:
            new_tariff = action['tariff']

        # New state
        new_state = {'load':new_load, 'soc':new_soc, 'tariff':new_tariff}
        return new_state

    def reward(self, state, action, new_state):
        """
        Inputs:
        Gets two dicts (state, action) and returns the new state of the system
        state: {'load': 0.4, 'soc': 1.0, 'tod': 2.0}
        action:  {'charge': -0.1}
        AND
        Returns: float value
        """
        # Here we compute the reward based on the state and the action we took
        tariff = new_state['tariff']
        price, price_im = self.tariff(tariff)
        return (action['procure'] * price) + (self.imbalance * price_im)

    def tariff(self, x_tariff):
        price = self.base_price + (self.pr_coeff * np.log(float(x_tariff) / float(self.max_tariff)))
        price_im = self.base_price - (self.pr_im_coeff * np.log(float(x_tariff) / float(self.max_tariff)))
        return price, price_im
