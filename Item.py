import numpy as np

class Item:
    __slots__ = ['fitness', 'values']
    def __init__(self, fitness=None, values=None):
        self.fitness = fitness
        self.values = values[:] if values is not None else None

    def initialize(self, lb=None, ub=None, size=None):
        self.values = np.random.uniform(lb, ub, size)
    
    def evaluate(self, solution):
        self.fitness = solution.eval(self.values)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return f'{self.fitness}'

    def __repr__(self):
        return f'{self.fitness} <- {self.values}'

    def __eq__(self, other):
        return self.fitness==other.fitness and (self.values == other.values).all()

    def __hash__(self):
        return hash(('fitness', self.fitness, 'values', self.values))