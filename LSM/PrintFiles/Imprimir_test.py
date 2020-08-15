import os
import pickle
from pathlib import Path

def impimir_1D_Encoding(spikes, labels, filename, filename2):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs(os.path.dirname(filename2), exist_ok=True)
    except FileNotFoundError:
        pass

    with open(filename, 'w') as of:
        for i, spike in enumerate(spikes):
            for j in spike:
                of.write(str(j) + " ")
            of.write('\n')
    of.close()

    with open(filename2, 'w') as of2:
        for i, l in enumerate(labels):
            of2.write(str(l))
            of2.write('\n')
    of2.close()


def leer_1D_Encoding(filename, filename2):
    spikes, labels = [], []

    if Path(filename).is_file():
        with open(filename) as file:
            for line in file:
                vector = [float(x) for x in line.split()]
                spikes.append(vector)

    if Path(filename2).is_file():
        with open(filename2) as file2:
            for line in file2:
                labels.append(int(line))

    return spikes, labels

def guardar_liquid(liquido, filename):
    with open(filename, 'wb') as out_s:
        pickle.dump(liquido, out_s)
    out_s.close()

def cargar_liquid(filename):
    with open(filename, 'rb') as in_s:
        liquido = pickle.load(in_s)
    return liquido

def impimirFtinessMulti(combi, variaces, filename):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    with open(filename, 'w') as of:
        for i, spike in enumerate(combi):
            of.write(str(spike) + " ")

        for i, spike in enumerate(variaces):
            of.write(str(spike) + " ")

    of.close()

def imprimirBestWeight(soluciones, filename):
    with open(filename, 'wb') as out_s:
        pickle.dump(soluciones, out_s)
    out_s.close()

def cargarBestWeight(filename):
    with open(filename, 'rb') as in_s:
        soluciones = pickle.load(in_s)
    return soluciones


