from liquidos.LiquidFeedForward import liquid_feedforward
from PrintFiles.Imprimir_test import leer_1D_Encoding, cargar_liquid, cargarBestWeight
from Fitness.MultiO import fitness
from Fitness.MonoO import fitness_Norton
from jmetal.util.solution import read_solutions



if __name__=='__main__':

    ruta = r"C:\Users\carlos\PycharmProjects\LSM\test"
    X_train, y_train = leer_1D_Encoding(str(ruta)+'\spikesTrain.txt', str(ruta)+'\labels_train.txt')
    X_test, y_test = leer_1D_Encoding(str(ruta)+'\spikesTest.txt', str(ruta)+'\labels_test.txt')

    timeSimulate = 15

    # Cargar liquido
    liquido = cargar_liquid(r"C:\Users\carlos\PycharmProjects\LSM\test\liquido.pickle")

    # Cargar los pesos y retardos optimizados
    soluciones = cargarBestWeight(r"C:\Users\carlos\PycharmProjects\LSM\test\PesosBest")

    liquido.update_W_D(soluciones[-1].variables)

    stateVectors = list()
    for p in X_train:
        liquido.reset()
        liquido.injectStimuli_1D(p)  # Estimulo 1D
        liquido.Simulate(timeSimulate)
        stateVectors.append(liquido.getLiquidState())

    print("[" + str(fitness(stateVectors, y_train)) + "]")

    stateVectors = list()
    for p in X_test:
        liquido.reset()
        liquido.injectStimuli_1D(p)  # Estimulo 1D
        liquido.Simulate(timeSimulate)
        stateVectors.append(liquido.getLiquidState())

    print("[" + str(fitness(stateVectors, y_test)) + "]")



