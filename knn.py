import numpy as np
from collections import Counter


class knn:

    def disEuclid(self, x1, x2):
        distance = 0
        for i in range(len(x1) - 1):
            if x1[i] is None and x2[i] is None:
                    distance += 1
            if x1[i]  is None and type(x2[i]) == str:
                distance += 1
            if x2[i] is None and type(x1[i]) == str:
                distance += 1
            if x1[i]  is None and x2[i] is not None and type(x2[i]) != str:
                distance += np.max(((x1[i] - x2[i]) ** 2),1-((x1[i] - x2[i]) ** 2))
            if x1[i] is not None and x2[i] is None and type(x1[i]) != str:
                distance +=np.max(((x1[i] - x2[i]) ** 2),1-((x1[i] - x2[i]) ** 2))
            if type(x2[i]) == str and type(x1[i]) == str:
                if x1[i] == x2[i]:
                    distance += 0
                else:
                    distance += 1
            if type(x2[i]) != str and type(x1[i]) != str:
                distance += ((x1[i] - x2[i]) ** 2)

        return np.sqrt(distance)

    def disManhattan(self, x1, x2):
        distance = 0
        for i in range(len(x1) - 1):
            distance += np.sum(np.abs(x1[i].astype(np.cfloat) - x2[i].astype(np.cfloat)))
        return distance

    def __init__(self, K=3):
        self.k = K

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        prediction = [self.__predict(x) for x in X]
        return prediction

    def __predict(self, x):
        # calculer la distance

        distances = [self.disEuclid(x, x_train) for x_train in self.X_train]

        # trouver les plus proches exemples
        k_indices = np.argsort(distances)[:self.k]
        k_proches = [self.y_train[i] for i in k_indices]

        # trouver la class la plus repeter
        class_majoritaire = Counter(k_proches).most_common()

        return class_majoritaire[0][0]

