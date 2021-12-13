import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class BA:
    def __init__(self, n_pop, bounds, n_iter, fmin=0, fmax=10, gamma=0.9, alpha=0.9):
        self.n_pop = n_pop
        self.bounds = np.array(bounds)
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.fmin = fmin
        self.fmax = fmax
        self.alpha = alpha
        self.gamma = gamma
        self.positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_pop, self.dim))
        self.velocity = np.zeros((self.n_pop, self.dim))
        self.f = np.zeros((self.n_pop, 1))
        self.A = np.random.uniform(1, 2, size=(self.n_pop, 1))
        self.r0 = np.random.uniform(0, 1, size=(self.n_pop, 1))
        self.r = np.copy(self.r0)
        self.best = None

    def update_position(self):
        self.beta = np.random.random((self.n_pop, 1))
        self.f = self.f + (self.fmax - self.fmin) * self.beta

        self.velocity += (self.positions - self.best) * self.f
        positions = self.positions + self.velocity
        positions = self.mapping(positions)
        positions = self.local_pos(positions)
        return positions

    def mapping(self, positions):
        # 边界检查
        cond1 = np.where(positions > self.bounds[:, 1], 0, 1)
        cond2 = np.where(positions < self.bounds[:, 0], 0, 1)
        self.velocity *= cond1 * cond2
        positions = np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])
        return positions

    def local_pos(self, positions):
        rnd = np.random.random((self.n_pop, 1))
        ind = np.where((rnd > self.r) > 0)[0]
        positions[ind] = self.best + np.random.uniform(-1, 1, 1) * self.A.mean()
        positions = self.mapping(positions)
        return positions

    def update_param(self, positions, count):
        rnd = np.random.random((self.n_pop, 1))
        fitness = np.apply_along_axis(objective, 1, positions)
        old_fitness = np.apply_along_axis(objective, 1, self.positions)
        cond1 = rnd < self.A
        cond2 = fitness < old_fitness
        ind = np.where((cond1.reshape(-1,1) * cond2) > 0)[0]
        if len(ind) > 0:
            self.positions[ind] = positions[ind]
            self.r[ind] = self.r0[ind] * (1 - np.exp(-self.gamma * count))
            self.A[ind, 0] = self.alpha * self.A[ind, -1]

    def update_best(self):
        fitness = np.apply_along_axis(objective, 1, self.positions)
        if self.best is None or min(fitness) < objective(self.best):
            self.best = np.copy(self.positions[np.argmin(fitness)])

    def run(self):
        count = 0
        self.update_best()
        self.cost = []
        while count < self.n_iter:
            positions = self.update_position()
            self.update_param(positions, count)
            self.update_best()
            self.cost.append(objective(self.best))
            count+=1


if __name__ == '__main__':
    ba = BA(n_pop=15, bounds=[[-5, 5]] * 2, n_iter=50)
    ba.run()

    import matplotlib.pyplot as plt

    plt.plot(ba.cost)
