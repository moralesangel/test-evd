import numpy as np

def crossval(x, y, k, i):

    assert k > 1
    assert i > 0
    assert i <= k
    assert len(x) == len(y)
    assert k <= len(y)

    print("K: ", k)
    print("I: ", i)

    n = len(y)
    size = n//k

    # Calcula los índices de inicio y fin para el conjunto de entrenamiento y prueba
    start_train = int(size * (i - 1))
    end_train = int(size * i)
    start_test = int(size * (i - 1))
    end_test = int(size * i)

    # Extrae los conjuntos de entrenamiento y prueba de x e y
    xtrn = np.concatenate((x[:start_train], x[end_train:]), axis=0)
    xtst = x[start_test:end_test]
    ytrn = np.concatenate((y[:start_train], y[end_train:]), axis=0)
    ytst = y[start_test:end_test]

    return xtrn, xtst, ytrn, ytst


if __name__ == "__main__":
    print("Crossval test")