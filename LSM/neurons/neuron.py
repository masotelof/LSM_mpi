from time import time

class SpikingNeuron:
    def __init__(self, spike=None, neuType='exc'):
        self.spike = spike
        self.neuType = 1 if neuType == 'exc' else -1

    def setSpike(self, spike):
        self.spike = spike

    def getSpike(self):
        return self.spike

    def getType(self):
        return self.neuType


class SpikeResponseModel(SpikingNeuron):
    def __init__(self, psp, thres=1, neuType='exc', position=None):
        SpikingNeuron.__init__(self, None, neuType)
        self.synapseSet = None
        self.psp = psp
        self.thres = thres
        self.neurons = None
        self.v = 0
        self.a = 0
        self.position = position

    def setSynapses(self, synapseSet):
        self.synapseSet = synapseSet

    def getSynapses(self):
        return self.synapseSet

    def setNeighbors(self, neurons):
        self.neurons = neurons

    def Voltage_empty(self, v=0):
        self.v = v

    def getPosition(self):
        return self.position

    def simulate(self, t):
        if self.spike is None:
            '''
            for key in self.synapseSet.keys():
                if self.neurons[key].getSpike() is not None:
                    w, d = self.synapseSet[key].getSynapse()
                    self.a += (w * self.neurons[key].getType()) * self.psp.evaluate(t - self.neurons[key].getSpike() - d)
                    # self.v += (w * self.neurons[key].getType()) * self.psp.evaluate(t-self.neurons[key].getSpike()-d)
            '''
            self.v += sum([(self.synapseSet[key].getSynapse()[0] * self.neurons[key].getType()) * self.psp.evaluate(t-self.neurons[key].getSpike()-self.synapseSet[key].getSynapse()[1]) for key in self.synapseSet if self.neurons[key].getSpike() is not None])

            # Comprobar al final de la sumatoria
            if self.v >= self.thres:
                self.spike = t
