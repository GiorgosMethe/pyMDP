import operator
import random
import numpy as np
import pickle


class Learner:
    def __init__(self, domain, alpha, gamma, maximization = True):
        self.domain = domain
        self.alpha = alpha
        self.gamma = gamma
        self.maximization = maximization

class QLearning(Learner):

    def __init__(self, domain, alpha, gamma, maximization = True):
        Learner.__init__(self, domain, alpha, gamma, maximization)
        self.Q = domain.initial_value * np.ones((domain.stateRange, domain.actionRange), dtype=float)

    def dumpLearning(self, name):
        pck_file = open( name + ".p", "wb" )
        pickle.dump(self.Q, pck_file)
        pck_file.close()

    def loadLearning(self, name):
        self.Q = pickle.load( open( name + ".p", "rb" ))

    def reset(self):
        self.Q = self.domain.initial_value * np.ones((self.domain.stateRange, self.domain.actionRange), dtype=float)

    def epsilonGreedy(self, state, epsilon): # return action
        if random.random() < epsilon: return self.domain.randomAction(state)
        else: return self.getOptimalAction(state)

    def getOptimalAction(self, state): # return best action for state
        s = self.domain.getStateFromRealValues(state)
        if self.maximization: return self.domain.getActionFromIndex(np.argmax(self.Q[s, :]))
        else: return self.domain.getActionFromIndex(np.argmin(self.Q[s, :]))

    def update(self, s, a, s_prime, reward):
        s = self.domain.getStateFromRealValues(s)
        s_prime = self.domain.getStateFromRealValues(s_prime)
        a = self.domain.getActionFromRealValues(a)
        ofv = 0.0
        if self.maximization: ofv = np.max(self.Q[s_prime,:]) # optimal future value
        else: ofv = np.min(self.Q[s_prime,:]) # optimal future value
        self.Q[s,a] = self.Q[s, a] + self.alpha * (reward + (self.gamma * ofv) - self.Q[s,a])

class SARSA(QLearning):

    def __init__(self, domain, alpha, gamma, maximization = True):
        QLearning.__init__(self, domain, alpha, gamma, maximization)

    def update(self, s, a, s_prime, a_prime, reward):
        if s is not None:
            s = self.domain.getStateFromRealValues(s)
            s_prime = self.domain.getStateFromRealValues(s_prime)
            a = self.domain.getActionFromRealValues(a)
            a_prime = self.domain.getActionFromRealValues(a_prime)
            self.Q[s,a] = self.Q[s, a] + self.alpha * (reward + (self.gamma * self.Q[s_prime,a_prime]) - self.Q[s,a])
