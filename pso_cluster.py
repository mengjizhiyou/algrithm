# Clustering using Particle Swarm Optimization
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
from pso import particle

######################################################################
"""original"""
plt.figure(figsize=(12, 5))
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state, centers=5)
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
"""pso cluster"""
def objective(centroids, labels, data):
    error = 0.0
    for i, c in enumerate(centroids):
        idx = np.where(labels == i)[0]
        dist = np.linalg.norm(data[idx] - c, axis=1).sum()
        dist /= len(idx)
        error += dist
    error /= len(centroids)
    return error


class Particle:
    def __init__(self, n_cluster, X, w=0.9, c1=0.5, c2=0.3):
        """
        self.position_i:表示多个聚类中心,这里是3个聚类中心(鸟群)
        """
        # super(particle, self).__init__() # 保持父的 __init__() 函数的继承
        index = np.random.choice(len(X), n_cluster)
        self.position_i = X[index].copy()
        self.pos_best_i = self.position_i.copy()
        self.velocity_i = np.zeros_like(self.position_i)
        self.best_obj = objective(self.position_i, self._predict(X), X)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_velocity(self, gbest):
        """Update velocity based on old value, cognitive component, and social component
        """
        v_old = self.w * self.velocity_i
        cognitive_component = self.c1 * np.random.random() * (self.pos_best_i - self.position_i)
        social_component = self.c2 * np.random.random() * (gbest - self.position_i)
        self.velocity_i = v_old + cognitive_component + social_component

    def update_position(self, X):
        """update the particle position based off new velocity updates"""
        self.position_i = self.position_i + self.velocity_i
        new_obj = objective(self.position_i, self._predict(X), X)
        if new_obj < self.best_obj:
            self.best_obj = new_obj
            self.pos_best_i = self.position_i.copy()

    def _predict(self, X):
        """Predict new data's cluster using minimum distance to centroid"""
        distance = self._calc_distance(X)
        cluster = self._assign_cluster(distance)
        return cluster

    def _calc_distance(self, X):
        """Calculate distance between data and centroids"""
        distances = []
        for c in self.position_i:
            distance = np.sum((X - c) * (X - c), axis=1)
            distances.append(distance)

        distances = np.array(distances)
        distances = np.transpose(distances)
        return distances

    def _assign_cluster(self, distance):
        """Assign cluster to data based on minimum distance to centroids"""
        cluster = np.argmin(distance, axis=1)
        return cluster


class ParticleCluster:
    def __init__(self, n_cluster, n_particles, X, max_iter=50, verbose=False):
        self.n_cluster = n_cluster
        self.n_particles = n_particles
        self.X = X
        self.max_iter = max_iter
        self.swarm = []
        self.gbest_obj = np.inf
        self.gbest = None
        self.cluster = None
        self.verbose = verbose

        for i in range(n_particles):
            part = Particle(self.n_cluster, self.X)
            if part.best_obj < self.gbest_obj:
                self.gbest = part.position_i.copy()
                self.gbest_obj = part.best_obj
                self.cluster = part._predict(self.X)
            self.swarm.append(part)

    def iter(self):
        print('initial global best obj: ', self.gbest_obj)
        fitness = []
        for i in range(self.max_iter):
            for part in self.swarm:
                part.update_velocity(self.gbest)
                part.update_position(self.X)
            for part in self.swarm:
                if part.best_obj < self.gbest_obj:
                    self.gbest = part.position_i
                    self.gbest_obj = part.best_obj
            fitness.append(self.gbest_obj)
            if self.verbose:
                if i % 10 == 0:
                    print('Iteration {:04d}/{:04d} current gbest score {:.18f}'.format(
                        i + 1, self.max_iter, self.gbest_obj))

        return fitness


n_cluster = 3
pso = ParticleCluster(
    n_cluster=n_cluster, n_particles=10, X=X)
fitness = pso.iter()
y_pred = pso.cluster
plt.subplot(133)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("PSO")

plt.figure()
plt.plot(range(1, len(fitness) + 1), fitness)
plt.xlabel('iterators');
plt.ylabel('fitness');

print('PSO centroids:\n',pso.gbest,'\n\n', 'Kmeans centroids:\n',model.cluster_centers_)
