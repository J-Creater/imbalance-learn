# @Project ：Experiment 
# @File    ：GDO.py
# @IDE     ：PyCharm 
# @Author  ：JYF
# @Date    ：2020/10/12 10:07

import numpy as np
import random
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from collections import Counter


def add_label(X, y):
    data = np.insert(X, X.shape[1], y, axis=1)  # 合并标签
    return data


class GDO:
    def __init__(self, k=5, alpha=1):
        self.k = k  # the number of neighbors
        self.alpha = alpha  # covariance coefficient
        self.N = 0  # the number of minority samples
        self.M = 0  # the number of majority samples
        self.l = 0  # the dimension of input data
        self.min_index = []  # index of minority samples
        self.maj_index = []  # index of majority samples

    def normalize(self, a):
        a = a.reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()  # normalize to 0-1
        a = min_max_scaler.fit_transform(a)
        return a.reshape(1, -1)[0]

    def Minority_instance_weighting(self, X, y, dist, indices):
        C = np.zeros(self.N)  # the density factor
        D = np.zeros(self.N)  # the distance factor
        I = np.zeros(self.N)  # the information weight

        for i, index in enumerate(self.min_index[0]):
            neigh_label = y[indices[index, 1:self.k + 1]]
            K_Ni_maj = Counter(neigh_label)[1]
            C[i] = K_Ni_maj / self.k

            neigh_maj_index = np.where(neigh_label == 1)[0] + 1  # index of majority nn
            dist_to_NN_all = sum(dist[index])  # the distance from Xi to instances of Ni
            dist_to_NN_maj = sum(dist[index, neigh_maj_index])  # the distance from Xi to instances of Ni_maj
            D[i] = dist_to_NN_maj / dist_to_NN_all

        I = C + D  # the information weight

        return self.normalize(I)

    def Probabilistic_anchor_instance_selection(self, I):
        a = [i for i in range(self.N)]  # 少数类样本 的 序号
        gamma = random.choices(a, weights=I, k=1)[0]  # ROULETTE SELECTION FOR THE MINORITY INSTANCES

        return gamma

    def New_instance_generation(self, I, min_sample):
        k = 1
        G = self.M - self.N  # samples need to generate
        new_instances = []

        neigh = NearestNeighbors(n_neighbors=2).fit(min_sample)
        dist_min, indices_min = neigh.kneighbors(min_sample)

        while k <= G:
            selected_index = self.Probabilistic_anchor_instance_selection(I)
            anchor = min_sample[selected_index]
            V = np.random.uniform(-1, 1, size=(1, self.l))[
                0]  # Randomly select a direction originating from the anchor minority instance

            d_0 = np.linalg.norm(anchor - V)
            mu = 0
            sigma = dist_min[selected_index, 1]  # the distance between anchor and its k-nearest minority neighbors
            d_i = self.alpha * sigma * np.random.randn(
                1) + mu  # d_i is a random number generated based on the Gaussian distribution
            r = d_i / d_0

            synthetic_instance = anchor + r * (V - anchor)
            new_instances.append(synthetic_instance)

            k += 1
        return np.array(new_instances)

    def fit_sample(self, X, y):
        self.min_index = np.where(y == 0)
        self.maj_index = np.where(y == 1)

        min_sample = X[self.min_index]
        maj_sample = X[self.maj_index]

        self.N = len(min_sample)
        self.M = len(maj_sample)
        self.l = X.shape[1]

        neigh = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        dist, indices = neigh.kneighbors(X)

        I = self.Minority_instance_weighting(X, y, dist, indices)

        new_instances = self.New_instance_generation(I, min_sample)

        Resampled_Data = np.concatenate((add_label(X, y), add_label(new_instances, 0)), axis=0)

        return Resampled_Data[:, :-1], Resampled_Data[:, -1]


# X, y = GDO(k=5, alpha=1).fit_sample(X, y)
