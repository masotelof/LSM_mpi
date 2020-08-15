import numpy as np


class Temporal_Intervals:
    def __init__(self, column, a, b):
        self.a = a
        self.b = b
        self.m = np.min(column)
        self.M = np.max(column)
        self.r = self.M - self.m

    def enconding_var(self, f):
        return (((self.b - self.a) / self.r) * f) + (((self.a * self.M) - (self.b * self.m)) / self.r)


class oneD_Encoding_set:

    def __init__(self, dataset, a, b):
        '''
        self.oneD = list()
        self.encondeX = list()

        for i in range(dataset.shape[1]):
            self.oneD.append(Temporal_Intervals(dataset[:, i], a, b))
        '''
        self.oneD = [Temporal_Intervals(dataset[:, i], a, b) for i in range(dataset.shape[1])]
        self.encondeX = list()

    def encoding(self, X):
        '''
        self.encondeX = list()

        for f in range(X.shape[0]):
            self.encondeX.append(self.oneD[f].enconding_var(X[f]))

        return self.encondeX
        '''
        return [self.oneD[f].enconding_var(X[f]) for f in range(X.shape[0])]