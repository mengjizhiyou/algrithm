import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class FA:
    def __init__(self, n_pop, n_iter, bounds, beta_0=1, alpha=0.2, beta_min=0.3, gamma=1):
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.bounds = bounds
        self.dim = len(bounds)
        self.beta_0 = beta_0
        self.beta_min = beta_min
        self.alpha = alpha
        self.gamma = gamma

    def init_pop(self):
        self.positions = np.zeros((self.n_pop, self.dim + 1))
        for i in range(self.n_pop):
            for j in range(self.dim):
                self.positions[i, j] = np.random.uniform(self.bounds[j][0], self.bounds[j][1])
            self.positions[i, -1] = objective(self.positions[i, :-1])

    def bright(self, x, y):
        rij = self._cal_distance(x, y)
        I_x = x[-1] * np.exp(-self.gamma * rij)
        I_y = y[-1] * np.exp(-self.gamma * rij)
        return I_x, I_y

    def attractiveness(self, x, y):
        rij = self._cal_distance(x, y)
        beta = self.beta_min + (self.beta_0 - self.beta_min) * np.exp(-self.gamma * (rij ** 2))
        return beta

    def _cal_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def update_position(self, x, y):
        for j in range(self.dim):
            x[j] = np.clip(x[j] + self.attractiveness(x, y) * (y[j] - x[j])
                           + self.alpha * (np.random.uniform() - 1 / 2),
                           self.bounds[j][0], self.bounds[j][1])
        x[-1] = objective(x[:-1])
        return x

    def run(self):
        count = 0
        self.cost = []
        self.init_pop()
        while count < self.n_iter:
            for i in range(self.n_pop):
                for j in range(self.n_pop):
                    if i != j:
                        """
                        将亮度看作目标值进行位置更新
                        # firefly_i = np.copy(self.positions[i, :])
                        # firefly_j = np.copy(self.positions[i, :])
                        # I_i, I_j = self.bright(firefly_i, firefly_j)
                        # if I_i < I_j:
                        #     self.positions[i, :] = np.copy(self.update_position(firefly_i, firefly_j))
                        # elif I_i > I_j:
                        #     self.positions[j, :] = np.copy(self.update_position(firefly_j, firefly_i))
                        """

                        # 根据目标函数进行更新
                        if self.positions[i, -1] > self.positions[j, -1]:
                            self.positions[i, :] = np.copy(
                                self.update_position(self.positions[i].copy(), self.positions[j].copy()))

            count += 1
            self.best = np.copy(self.positions[self.positions[:, -1].argsort()[-1]])
            self.cost.append(self.best[-1])


if __name__ == '__main__':
    fa = FA(n_pop=20, n_iter=20, bounds=[[-5, 5]] * 2, beta_0=1, alpha=0.2, beta_min=0.3, gamma=1)
    fa.run()

    # import matplotlib.pyplot as plt
    # plt.plot(fa.cost)
