import numpy as np


class psp:
    def __init__(self, tau=9):
        self.tau = tau

    def evaluate(self, t):
        '''
        resp = 0
        if t > 0:
            resp = (t / self.tau) * np.exp(1 - (t / self.tau))
        return resp
        '''
        return (t / self.tau) * np.exp(1 - (t / self.tau)) if t > 0 else 0
