import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class CEM:
    def __init__(self, num, bounds, n_iter, alpha=0.7, rho=0.5):
        self.num = num
        self.bounds = np.array(bounds)
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.alpha = alpha
        self.rho = rho
        self.Ne = int(self.rho * self.num)
        self.best = None

    def init_variable(self):
        """随机初始化"""
        self.var = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.num, self.dim))
        self.mean, self.std = np.mean(self.var, axis=0), np.std(self.var, axis=0)

    def gen_sample(self):
        """生成样本"""
        loss = np.apply_along_axis(objective, 1, self.var)
        self.var = self.var[loss.argsort()]
        self.var[self.Ne:, :] = np.random.normal(self.mean, self.std, (self.num - self.Ne, self.dim))
        self.var = np.clip(self.var, self.bounds[:, 0], self.bounds[:, 1])

    def update_distribution(self):
        loss = np.apply_along_axis(objective, 1, self.var)
        self.var = self.var[loss.argsort()]
        for j in range(self.dim):
            self.mean[j] = self.alpha * self.mean[j] + (1 - self.alpha) * self.var[:self.Ne, j].mean()
            self.std[ j] = self.alpha * self.std[j] + (1 - self.alpha) * self.var[:self.Ne, j].std()
            if self.std[j] < 0.005: self.std[j] = 3

    def update_best(self):
        loss = np.apply_along_axis(objective, 1, self.var)
        if self.best is None:
            self.best = self.var[np.argmin(loss)]
        else:
            if objective(self.best) > loss.min():
                self.best = self.var[np.argmin(loss)]

    def run(self):
        self.init_variable()
        self.update_best()
        t = 0
        self.cost = []
        while t < self.n_iter:
            self.gen_sample()
            self.update_distribution()
            self.update_best()
            self.cost.append(objective(self.best))
            t += 1

if __name__ == '__main__':
    cem = CEM(num=30, bounds=[[-5, 5]] * 2, n_iter=20)
    cem.run()

    import matplotlib.pyplot as plt

    plt.plot(cem.cost)
