
class StaticSynapse:

    def __init__(self, weight, delay):
        self.weight = weight
        self.delay = delay

    def setSynapse(self, weight, delay):
        self.weight, self.delay = weight, delay

    def getSynapse(self):
        return self.weight, self.delay
