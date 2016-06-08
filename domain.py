import numpy as np
import itertools as it
import random
import decimal

class Variable(object):
    def __init__(self, min, max, step = None, n_dim = 1):
        self.min = min
        self.max = max
        self.step = step
        self.continuous = (step == None)
        self.n_dim = n_dim
        self.space = np.arange(min, max+0.0000001, step)
        if n_dim > 1:
            self.space = np.array(list(it.product(self.space, repeat = n_dim)))
        self.range = range(len(self.space))

    def rangeToRealValue(self, range):
        if self.n_dim == 1:
            return np.round(self.space[range],abs(decimal.Decimal(str(self.step)).as_tuple().exponent))
        else:
            return np.round(self.space[range],abs(decimal.Decimal(str(self.step)).as_tuple().exponent))

    def realValueToRange(self, real):
        if self.n_dim == 1:
            return int(round((round(real,abs(decimal.Decimal(str(self.step)).as_tuple().exponent)) - self.min) / self.step))
        else:
            all_states = len(self.range)
            index = 0
            one = len(np.arange(self.min, self.max+0.0000001, self.step))
            for dim in real:
                all_states = all_states / one
                index = index + (all_states * round((round(dim,abs(decimal.Decimal(str(self.step)).as_tuple().exponent)) - self.min) / self.step))
            return index

class Domain(object):
    def __init__(self):
        print 'Initializing domain'
        self.simulated_step = 0
        self.state_var = []
        self.action_var = []
        self.stateSpace = self.stateRange = None
        self.actionSpace = self.actionRange = None

    def simulate(self, state, action):
        self.simulated_step = self.simulated_step + 1

    def randomAction(self, state):
        if (type(state) != int): state = self.getStateFromRealValues(state) # if not state index
        action = random.randint(0,self.actionRange-1)
        return self.getActionFromIndex(action)

    def setStateVariables(self, stateDictionary):
        print "-Building state space"
        self.state_variables = stateDictionary
        self.stateSpace, self.stateRange = self.buildSpace(self.state_variables, self.state_var)
        print '-- total:', self.stateRange, 'states'

    def getStateFromRealValues(self, real_values):
        state_index = 0
        all_states = self.stateRange
        for var_name in self.state_var:
            all_states = all_states / len(self.state_variables[var_name].space)
            state_index = state_index + (self.state_variables[var_name].realValueToRange(real_values[var_name]) * all_states)
        return state_index

    def getStateFromIndex(self, state_index):
        index = 0
        real_values = {}
        for name in self.state_var:
            real_values[name] = self.state_variables[name].rangeToRealValue(self.stateSpace[state_index][index])
            index = index + 1
        return real_values

    def getActionFromRealValues(self, real_values):
        action_index = 0
        all_states = self.actionRange
        for var_name in self.action_var:
            all_states = all_states / len(self.action_variables[var_name].space)
            action_index = action_index + (self.action_variables[var_name].realValueToRange(real_values[var_name]) * all_states)
        return action_index

    def getActionFromIndex(self, action_index):
        index = 0
        real_values = {}
        for name in self.action_var:
            real_values[name] = self.action_variables[name].rangeToRealValue(self.actionSpace[action_index][index])
            index = index + 1
        return real_values

    def buildSpace(self, variables, state_var):
        space = []
        for name, var in variables.iteritems():
            print '-- variable:', '(', name, ')', 'with:', len(var.range), "states"
            space.append(var.range)
            state_var.append(name)
            stateSpace = list(it.product(*space))
        return stateSpace, len(stateSpace)

    def setActionVariables(self, stateDictionary):
        print "-Building action space"
        self.action_variables = stateDictionary
        self.actionSpace, self.actionRange = self.buildSpace(self.action_variables, self.action_var)
        print '-- total:', self.actionRange, 'actions'

    def transition(self):
        return self.transition()
