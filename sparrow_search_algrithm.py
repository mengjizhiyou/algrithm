import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class SSA:
    def __init__(self, n_pop, bounds, n_iter, p_ratio=0.2, w_ratio=0.1, ST=0.8, epsilon=1e-2):
        self.n_pop = n_pop
        self.bounds = np.array(bounds)
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.n_producer = int(n_pop * p_ratio)
        self.positions = self.init_pop()
        self.best = None  # 全局最优
        self.worst = None  # 全局最差
        self.ST = ST
        self.epsilon = epsilon
        self.p_best = None  # 生产者的最佳位置
        self.n_warning = int(self.n_pop * w_ratio)

    def init_pop(self):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.n_pop, self.dim))
        return pop

    def update_fitness(self):
        self.fitness = np.apply_along_axis(objective, 1, self.positions)
        self.f_sort = np.argsort(self.fitness)
        min_ = np.argmin(self.fitness)
        max_ = np.argmax(self.fitness)
        if self.best is None:
            self.best = np.copy(self.positions[min_])
            self.worst = np.copy(self.positions[max_])
        else:
            if objective(self.best) > objective(self.positions[min_]):
                self.best = np.copy(self.positions[min_])
            if objective(self.worst) < objective(self.positions[max_]):
                self.worst = np.copy(self.positions[max_])

    def mapping(self, val):
        return np.clip(val, self.bounds[:, 0], self.bounds[:, 1])

    def update_p(self, producer):
        p_fitness = np.apply_along_axis(objective, 1, producer)
        self.p_best = producer[np.argmin(p_fitness)].copy()

    def update_producer(self):
        r2 = np.random.rand(1)
        index_p = self.f_sort[:self.n_producer]
        if r2 < self.ST:  # 预警值较小，没有捕食者出现
            alpha = np.random.uniform(size=(self.n_producer, self.dim))
            i = np.arange(self.n_producer).reshape(-1, 1)
            self.positions[index_p] = self.positions[index_p] * np.exp(-i / (alpha * self.n_iter))
        else:
            Q = np.random.normal(loc=0, scale=1.0, size=(self.n_producer, self.dim))
            L = np.ones((1, self.dim))
            self.positions[index_p] = self.positions[index_p] + Q * L
        self.positions = self.mapping(self.positions)
        self.update_p(self.positions[index_p])

    def update_follwer(self):

        # 这一部分追随者是围绕最好的发现者周围进行觅食，其间也有可能发生食物的争夺，使其自己变成生产者
        index_f = self.f_sort[self.n_producer:self.n_pop // 2]
        A = np.random.choice([-1, 1], size=(len(index_f), self.dim))
        A_plus = A.T * (1. / (A * A).sum(axis=1))
        L = np.ones((1, self.dim))
        self.positions[index_f] = self.p_best + np.abs(self.positions[index_f] - self.p_best)*A_plus.T * L

        # 这一部分麻雀处于十分饥饿的状态（因为它们的能量很低，也就是适应度值很差），需要到其它地方觅食
        index_h = self.f_sort[self.n_pop // 2:]
        i = np.arange(self.n_pop // 2, self.n_pop).reshape(-1, 1)
        Q = np.random.normal(loc=0, scale=1.0, size=(len(index_h), self.dim))
        self.positions[index_h] = Q * np.exp((self.worst - self.positions[index_h]) / np.square(i))
        self.positions = self.mapping(self.positions)

    def update_warning(self):
        """这一部位为意识到危险（注意这里只是意识到了危险，不代表出现了真正的捕食者）的麻雀的位置更新"""
        # 处于种群外围的麻雀向安全区域靠拢
        c = np.random.permutation(np.arange(self.n_pop))
        b = self.f_sort[c[:self.n_warning]]
        f_gt_g = np.argwhere((self.fitness[b] > objective(self.best)) > 0)
        if len(f_gt_g)>0:
            beta = np.random.normal(size=(len(f_gt_g), self.dim))
            self.positions[f_gt_g] = self.best + beta * np.abs(self.positions[f_gt_g] - self.best)

        # 处在种群中心的麻雀则随机行走以靠近别的麻雀
        f_eq_g = np.argwhere((self.fitness[b] <= objective(self.best)) > 0)
        if len(f_eq_g)>0:
            k = np.random.uniform(-1, 1, size=(len(f_eq_g), self.dim))
            self.positions[f_eq_g] = self.positions[f_eq_g] + k * np.abs(self.positions[f_eq_g] - self.worst) / \
                                     (self.fitness[f_eq_g] - objective(self.worst) + self.epsilon)
        self.positions = self.mapping(self.positions)

    def run(self):
        count = 0
        self.cost = []
        self.update_fitness()
        while count < self.n_iter:
            self.update_producer()
            self.update_follwer()
            self.update_warning()
            self.update_fitness()
            self.cost.append(objective(self.best))
            count += 1


if __name__ == '__main__':
    ssa = SSA(n_pop=15, bounds=[[-5, 5]] * 2, n_iter=50)
    ssa.run()

    import matplotlib.pyplot as plt

    plt.plot(ssa.cost)
