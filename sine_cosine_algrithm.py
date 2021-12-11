import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class SCA:
    def __init__(self, n_pop, bounds, n_iter, a=2, r1=2):
        self.n_pop = n_pop
        self.bounds = np.array(bounds)
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.best = None
        self.a = a
        self.r1 = r1
        self.best = None

    def init_variable(self, size):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (size, self.dim))
        return pop

    def update_position(self):
        r2 = 2 * np.pi * np.random.uniform(0, 1, (self.n_pop, self.dim))
        r3 = 2 * np.random.uniform(0, 1, size=(self.n_pop, self.dim))
        r4 = np.random.uniform(0, 1, size=(self.n_pop, self.dim))
        r4_sin = np.where(r4 < 0.5, 1, 0) * np.sin(r2)
        r4_cos = np.where(r4 >= 0.5, 1, 0) * np.cos(r2)
        self.positions = self.positions + self.r1 * (r4_sin + r4_cos) * abs(r3 * self.p - self.positions)
        self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])

    def update_p(self):
        fitness = np.apply_along_axis(objective, 1, self.positions)
        self.p = np.copy(self.positions[np.argmin(fitness)])

    def run(self):
        self.positions = self.init_variable(self.n_pop)
        self.update_p()
        self.best = np.copy(self.p)
        count = 0
        self.cost = []
        while count < self.n_iter:
            self.r1 = self.a - count * (self.a / self.n_iter)
            self.update_position()
            self.update_p()
            if objective(self.best) > objective(self.p):
                self.best = np.copy(self.p)
                self.cost.append(objective(self.best))
            count += 1


if __name__ == '__main__':
    sca = SCA(n_pop=15, bounds=[[-5, 5]] * 2, n_iter=200)
    sca.run()

    import matplotlib.pyplot as plt

    plt.plot(sca.cost)
