import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class SSA:
    def __init__(self, n_pop, bounds, n_iter):
        self.n_pop = n_pop
        self.bounds = np.array(bounds)
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.best = None

    def init_variable(self, size):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (size, self.dim))
        return pop

    def update_food(self):
        fittness = np.apply_along_axis(objective, 1, self.positions)
        min_ind = np.argmin(fittness)
        min_pos = self.positions[min_ind].reshape(1, -1)
        self.food_pos = np.copy(min_pos) if fittness[min_ind] < objective(self.food_pos[0]) else self.food_pos

    def update_leader(self, count):
        c1 = 2 * np.exp(-(4 * count / self.n_iter) ** 2)
        c2 = np.random.uniform(0, 1, size=(self.n_pop // 2, self.dim))
        c3 = np.random.uniform(0, 1, size=(self.n_pop // 2, self.dim))
        c3 = np.where(c3 >= 0.5, 1, -1)
        same_part = c2 * (self.bounds[:, 1].T - self.bounds[:, 0].T) + self.bounds[:, 0].T
        self.positions[:self.n_pop // 2] = self.food_pos + c3 * c1 * same_part

    def update_follwer(self):
        self.positions[self.n_pop // 2:] = (self.positions[self.n_pop // 2 - 1:self.n_pop-1] +
                                            self.positions[self.n_pop // 2:]) / 2
        self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])

    def run(self):
        self.positions = self.init_variable(size=self.n_pop)
        self.food_pos = self.init_variable(size=1)
        self.best = np.copy(self.food_pos)
        count = 0
        self.cost = []
        while count < self.n_iter:
            self.update_food()
            self.update_leader(count)
            self.update_follwer()
            if objective(self.best[0]) > objective(self.food_pos[0]):
                self.best = np.copy(self.food_pos)
                self.cost.append(objective(self.best[0]))
            count+=1


if __name__ == '__main__':
    ssa = SSA(n_pop=15, bounds=[[-5, 5]] * 2, n_iter=20)
    ssa.run()

    import matplotlib.pyplot as plt

    plt.plot(ssa.cost)
