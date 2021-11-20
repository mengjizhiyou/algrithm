import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

######################################################################
"""original"""
plt.figure(figsize=(12, 5))
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, n_features=6, random_state=random_state, centers=5)
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("original")

######################################################################
"""sklearn.cluster.Kmeans"""
model = KMeans(n_clusters=3, random_state=random_state)
y_pred = model.fit_predict(X)
plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("kmeans")

######################################################################
"""GA cluster"""


def objective(chromosome, labels, data):
    centroids = [gene.centroid for gene in chromosome]
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)[0]
        dist = np.linalg.norm(data[idx] - c, axis=1).sum()
        dist /= len(idx)
        error += dist
    error /= len(centroids)
    return error


class gene:
    def __init__(self, bound, dimensions, index, init_centroid=None):
        self.index = index
        if init_centroid is not None:
            self.centroid = init_centroid
        else:
            self.centroid = []
            for i in range(dimensions):
                self.centroid.append(random.uniform(bound[i][0], bound[i][1]))


class GACluster:
    def __init__(self, X, n_pop, n_iter, n_cluster, r_cross, r_mut, init_centroid=None):
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.n_cluster = n_cluster
        self.X = X
        self.r_cross = r_cross
        self.r_mut = r_mut
        self.bounds = [*zip(X.min(axis=0), X.max(axis=0))]
        self.n_dim = X.shape[1]
        self.pop = []
        self.init_centroid = init_centroid

    def init_pop(self):
        for i in range(self.n_pop):
            chromosome = []
            for j in range(self.n_cluster):
                chromosome.append(gene(self.bounds, self.n_dim, j))
            self.pop.append(chromosome)
        if self.init_centroid is not None:
            init_centroid_list = []
            for i in range(len(self.init_centroid)):
                init_centroid_list.append(
                    gene(self.bounds, self.n_dim, i, init_centroid=self.init_centroid[i]))
            self.pop[0] = init_centroid_list

    def fitness(self):
        fitness = np.zeros(self.n_pop)
        for i in range(self.n_pop):
            chromosome = self.pop[i]
            labels = self.predict(chromosome)
            fitness[i] = objective(chromosome, labels, self.X)
        return fitness

    def selection(self, fitness, k=2):
        """该函数获取种群并返回 1个选中的父母。
        k值表示从群体中选的个体数，固定为 2，用于父母选择"""
        selection_ix = np.random.randint(len(self.pop))
        for ix in np.random.randint(0, len(self.pop), k):
            if fitness[ix] < fitness[selection_ix]:
                selection_ix = ix
        return self.pop[selection_ix]

    def crossover(self, p1, p2):
        """根据交叉率生成后代"""
        # 将父代复制到下一代
        c1, c2 = p1.copy(), p2.copy()
        for i in range(len(p1)):
            # 抽取[0,1]范围内的随机数来确定是否执行crossover
            if np.random.rand() < self.r_cross:
                # 选择一个有效的分割点
                pt = np.random.randint(1, self.n_dim-2)
                # 执行crossover
                c1[i].centroid = p1[i].centroid[:pt] + p2[i].centroid[pt:]
                c1[i].centroid = p2[i].centroid[:pt] + p1[i].centroid[pt:]
        return [c1, c2]

    def mutation(self, chromosome):
        for i, gene in enumerate(chromosome):
            bitstring = gene.centroid
            for j in range(len(bitstring)):
                if np.random.rand() > self.r_mut:
                    bitstring[i] = random.uniform(self.bounds[j][0], self.bounds[j][1])  # 翻转位
            chromosome[i].centroid = bitstring

    def predict(self, chromosome):
        """Predict new data's cluster using minimum distance to centroid"""
        distance = self._calc_distance(self.X, chromosome)
        cluster = self._assign_cluster(distance)
        return cluster

    def _calc_distance(self, X, chromosome):
        """Calculate distance between data and centroids"""
        distances = []
        for gene in chromosome:
            distance = np.sum((X - np.array(gene.centroid)) * (X - np.array(gene.centroid)), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    def _assign_cluster(self, distance):
        """Assign cluster to data based on minimum distance to centroids"""
        cluster = np.argmin(distance, axis=1)
        return cluster

    def run(self):
        self.init_pop()
        chromosome = self.pop[0]
        labels = self.predict(chromosome)
        best, best_obj = 0, objective(chromosome, labels, self.X)
        for generation in range(self.n_iter):
            # 评估所有候选解
            fitness = self.fitness()
            for i in range(self.n_pop):
                if fitness[i] < best_obj:
                    best, best_obj = self.pop[i], fitness[i]
            # 调用selection函数n_pop次，以创建父母列表
            selected = [self.selection(fitness) for _ in range(self.n_pop)]
            # 循环遍历父级列表，并创建一个用作下一代的子级列表，根据需要调用交叉和变异函数。
            children = list()
            for i in range(0, self.n_pop, 2):
                # 选择成对的父母
                p1, p2 = selected[i], selected[i + 1]
                for c in self.crossover(p1, p2):
                    self.mutation(c)
                    children.append(c)
            self.pop = children  # 替换种群
        return [best, best_obj]


if __name__ == '__main__':
    n_iter = 300  # 总迭代数
    n_pop = 100  # 群体大小
    r_cross = 0.9  # 交叉率
    r_mut = 1.0 / (X.shape[1] * 3)  # 突变率
    n_cluster = 3
    ga = GACluster(X, n_pop, n_iter, n_cluster, r_cross, r_mut)
    best, best_obj = ga.run()
    y_pred = ga.predict(best)
    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("GA")
