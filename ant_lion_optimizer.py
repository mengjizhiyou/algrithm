import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class ALO:
    def __init__(self, n_pop, bounds, n_iter):
        self.n_pop = n_pop
        self.bounds = np.array(bounds)
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.best = None
        self.ant = self.init_variable(self.n_pop)
        self.antlion = self.init_variable(self.n_pop)
        self.w_const = np.array([0.1, 0.5, 0.75, 0.9, 0.95]) * self.n_iter

    def init_variable(self, size):
        pop = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (size, self.dim))
        return pop

    def cal_fitness(self, pos):
        return np.apply_along_axis(objective, 1, pos)

    def roulette(self):
        """利用轮盘赌轮盘算子根据蚁狮的适应度对蚁狮进行选择"""
        fitness = self.cal_fitness(self.antlion)
        fitness_norm = fitness / sum(fitness)
        rand = np.random.uniform(size=1)
        while (fitness_norm <= rand).sum() < 1:
            rand = np.random.uniform(size=1)
        index = np.argwhere(fitness_norm <= rand).flatten()[0]
        return index

    def update_c_d(self, count, pos):
        c = np.min(pos, axis=0)
        d = np.max(pos, axis=0)
        try:
            w = np.where(self.w_const < count)[0][0] + 1
            I = 10 ** w * count / self.n_iter
        except:
            I = 1  # 针对count=0的情况

        c /= I
        d /= I

        rand = np.random.uniform(size=pos.shape)
        rand = np.where(rand > 0.5, 1, -1)
        c_i = rand * (pos + c) + pos
        d_i = rand * (pos + d) + pos
        return c_i, d_i

    def update_ant(self, count):
        rand = np.random.uniform(size=self.ant.shape)
        rand = np.where(rand > 0.5, 1, 0)

        random_walk = 2 * rand - 1
        random_walk[:, 0] = 0
        random_walk = np.cumsum(random_walk, axis=1)
        min_i = np.min(random_walk, axis=1).reshape(-1, 1)
        max_i = np.max(random_walk, axis=1).reshape(-1, 1)

        # 每只蚂蚁根据轮盘赌选中的蚁狮随机游走
        random_antlion = [self.roulette() for _ in range(self.n_pop)]
        c_i, d_i = self.update_c_d(count, self.antlion[random_antlion])
        antlion_random = (random_walk - min_i) * (d_i - c_i) / (min_i - max_i) + c_i

        # 每只蚂蚁围绕精英蚁狮随机游走
        fitness = self.cal_fitness(self.antlion)
        elite = self.antlion[np.argmin(fitness)].copy()
        c_i, d_i = self.update_c_d(count, elite)
        elite_random = (random_walk - min_i) * (d_i - c_i) / (min_i - max_i) + c_i

        self.ant = (antlion_random + elite_random) / 2

    def update_antlion(self):
        pop = np.vstack([self.antlion, self.ant])
        pop = np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])
        fitness = self.cal_fitness(pop)
        index = np.argsort(fitness)
        self.antlion = pop[index[:self.n_pop]].copy()
        self.ant = pop[index[self.n_pop:]].copy()

    def run(self):
        self.best = self.antlion[np.argmin(self.cal_fitness(self.antlion))].copy()
        count = 0
        self.cost = []
        while count < self.n_iter:
            self.update_ant(count)
            self.update_antlion()
            fitness = self.cal_fitness(self.antlion)
            if objective(self.best) > fitness.min():
                self.best = self.antlion[np.argmin(fitness)].copy()
                self.cost.append(objective(self.best))
            count += 1


if __name__ == '__main__':
    alo = ALO(n_pop=10, bounds=[[-5, 5]] * 2, n_iter=20)
    alo.run()

    import matplotlib.pyplot as plt

    plt.plot(alo.cost)
