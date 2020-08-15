import numpy as np
import os

class LoadDataset:
    def __init__(self, ruta, name_dataset, num):
        '''
        self.ruta_spikesTrain = ruta + name_dataset + "\\Spikes_Train\\" + name_dataset + "_Train" + str(num)
        self.ruta_spikesTest = ruta + name_dataset + "\\Spikes_Test\\" + name_dataset + "_Test" + str(num)
        self.ruta_labelsTrain = ruta + name_dataset + "\\labels_Train\\labels_Train" + str(num)
        self.ruta_labelsTest = ruta + name_dataset + "\\labels_Test\\labels_Test" + str(num)
        '''
        self.ruta_spikesTrain = os.path.join(ruta, name_dataset, "Spikes_Train",  f'{name_dataset}_Train{num}')
        self.ruta_spikesTest = os.path.join(ruta, name_dataset, "Spikes_Test",  f'{name_dataset}_Test{num}')
        self.ruta_labelsTrain = os.path.join(ruta, name_dataset, "labels_Train", f'labels_Train{num}')
        self.ruta_labelsTest = os.path.join(ruta, name_dataset, "labels_Test", f'labels_Test{num}')
        self.spikesTrain = list()
        self.spikesTest = list()
        self.labelsTrain = list()
        self.labelsTest = list()

        with open(self.ruta_spikesTrain) as file:
            # for line in file:
            #     #vector = [float(x) for x in line.split()]
            #     self.spikesTrain.append([float(x) for x in line.split()])

            # self.spikesTrain2 = [line for line in file for x in line.split()]
            self.spikesTrain = [[float(x) for x in line.split()] for line in file]

        with open(self.ruta_spikesTest) as file:
            self.spikesTest = [[float(x) for x in line.split()] for line in file]

        with open(self.ruta_labelsTrain) as file:
            self.labelsTrain = [int(x) for line in file for x in line.split()]

        with open(self.ruta_labelsTest) as file:
            self.labelsTest = [int(x) for line in file for x in line.split()]

    def getSpikesTrain(self):
        return self.spikesTrain

    def getLabelsTrain(self):
        return self.labelsTrain

    def getSpikesTest(self):
        return self.spikesTest

    def getLabelsTest(self):
        return self.labelsTest
