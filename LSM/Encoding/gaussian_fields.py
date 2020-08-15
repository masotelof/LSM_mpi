#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 08:13:06 2019

@author: andres
"""

import math
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# import pyNN.nest as sim

class GaussianFunction:

    def __init__(self, mean, sd, A=1):
        self.mean = mean
        self.sd = sd
        self.A = A

    def evaluate(self, x):
        return self.A * math.exp(-math.pow(x - self.mean, 2) / (2 * math.pow(self.sd, 2)))


class GaussianReceptiveFields:

    def __init__(self, X, simTime, m=3, gamma=1.5, supress_late_spikes=True, A=1):
        self.X = X
        self.simTime = simTime
        n_min, n_max = np.min(self.X), np.max(self.X)
        self.m = m
        w = float(n_max - n_min) / (gamma * (self.m - 2))
        self.supress_late_spikes = supress_late_spikes
        '''
        self.gf = list()
        for i in range(1, self.m + 1):
            C = n_min + (float(2 * i - 3) / 2) * (float(n_max - n_min) / float(self.m - 2))
            self.gf.append(GaussianFunction(C, w, A))
        '''
        self.gf = [GaussianFunction(n_min + (float(2 * i - 3) / 2) * (float(n_max - n_min) / float(self.m - 2)), w, A) for i in range(1, self.m + 1)]
        self.encodedVar = list()

    def encoding_var(self, x):
        self.encodedVar = list()
        for g in self.gf:
            T = (1 - g.evaluate(x)) * self.simTime
            ft = list()
            if not (self.supress_late_spikes and T > (self.simTime * 0.9)):
                ft.append(T + 1.1)
            self.encodedVar.append(ft)

        return self.encodedVar


class GRF_Set:

    def __init__(self, X, simTime, m=3, gamma=1.5, supress_late_spikes=True, A=1):
        self.encodedX = list()
        '''
        self.grf = list()
        for f in range(X.shape[1]):
            self.grf.append(GaussianReceptiveFields(X[:, f], simTime, m, gamma, supress_late_spikes))
        '''
        self.grf = [GaussianReceptiveFields(X[:, f], simTime, m, gamma, supress_late_spikes) for f in range(X.shape[1])]

    def encoding(self, X):
        '''
        self.encodedX = list()

        for f in range(X.shape[0]):
            self.encodedX.append(self.grf[f].encoding_var(X[f]))

        return self.encodedX
        '''
        return [self.grf[f].encoding_var(X[f]) for f in range(X.shape[0])]

    # if __name__ == "__main__":
#     X, y = datasets.load_iris(return_X_y = True)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
#     m = 4
#     encoder = GRF_Set(X_train, 10, m, supress_late_spikes=False)
#     tmp = np.array([4.9,3.6,1.4,0.1])
#     spike_trains = encoder.encoding(tmp)

#     sim.setup()
#     inputs = list()
#     for sts  in spike_trains:
#         pop = sim.Population(m, sim.SpikeSourceArray, {'spike_times': sts})
#         print (sts)
#         print ("*****")
#         print (pop.describe())
#     sim.end()