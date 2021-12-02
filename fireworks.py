# refer to https://github.com/GoodLittleStar/Fireworks


import numpy as np
import random
import copy


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class FireWork:
    """单个烟火"""
    num = 0

    def __init__(self, dims, bound):
        self.dims = dims
        self.bound = bound
        self.fitness = 0
        self.Si = 0
        self.Ai = 0
        self.Ri = 0

    def init_position(self):
        self.position = [random.uniform(self.bound[j][0], self.bound[j][1]) for j in range(self.dims)]

    def cal_fitness(self):
        self.fitness = objective(self.position)
        FireWork.num += 1

    def cal_dist(self, other):
        return np.sqrt(np.sum(np.square(np.array(self.position) - np.array(other.position))))


class FWA:
    """多个烟火"""

    epsilon = 1e-30

    def __init__(self, n_pop, n_iter, initbounds, realbounds, a, b, A_hat, m, n_mut):
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.initbounds = initbounds
        self.realbounds = realbounds
        self.Y_max = 0  # Y_max
        self.Y_min = 0  # Y_min
        self.dims = len(initbounds)
        self.a = a
        self.b = b
        self.A_hat = A_hat
        self.m = m
        self.n_mut = n_mut  # 变异火花数
        self.best = None

    def init_pop(self):
        self.positions = []

        for i in range(self.n_pop):
            position = FireWork(self.dims, self.initbounds)
            position.init_position()
            position.cal_fitness()
            self.positions.append(position)

        self.update_fitness()
        self.best = self.positions[self.best_ind]

    def update_fitness(self, type='min'):
        """更新最大、最小适应度"""

        fitnesses = [self.positions[i].fitness for i in range(len(self.positions))]

        if type == 'min':
            self.Y_min = min(fitnesses)
            self.best_ind = np.argmin(fitnesses)
        else:
            self.Y_max = max(fitnesses)
            self.worst_ind = np.argmax(fitnesses)

    def explosion(self):
        self._create_spark()
        self._explosion_range()

        for i in range(self.n_pop):
            new_positions = []

            # 对每个火花进行位移操作
            for k in range(self.positions[i].Si):
                spark = copy.deepcopy(self.positions[i])

                # 随机维度进行位移
                rnd_dim_num = round(self.dims * random.random())
                rnd_dim = random.sample(range(self.dims), rnd_dim_num)

                # 位移
                for j in rnd_dim:
                    spark.position[j] += self._shift(spark.position[j], spark.Ai)
                    self.mapping(spark, j)

                spark.cal_fitness()
                new_positions.append(spark)

            self.positions += new_positions

    def _create_spark(self):
        """产生火花，计算每个烟花的火花数"""
        self.update_fitness(type='max')
        diff = sum([self.Y_max - self.positions[i].fitness for i in range(self.n_pop)])

        for i in range(self.n_pop):
            Si = self.m * (self.Y_max - self.positions[i].fitness + self.epsilon) \
                 / (diff + self.epsilon)
            if Si < self.a * self.m:
                self.positions[i].Si = round(self.a * self.m)
            elif Si > self.b * self.m:
                self.positions[i].Si = round(self.b * self.m)
            else:
                self.positions[i].Si = round(Si)

    def _explosion_range(self):
        self.update_fitness()
        diff = sum([self.Y_min - self.positions[i].fitness for i in range(self.n_pop)])
        for i in range(self.n_pop):
            self.positions[i].Ai = self.A_hat * (self.positions[i].fitness - self.Y_min + self.epsilon) \
                                   / (diff + self.epsilon)

    def _shift(self, x_j, Ai):
        x_delta = x_j + random.uniform(0, Ai)
        return x_delta

    def mutation(self):
        new_positions = []
        for i in range(self.n_mut):
            rnd = random.randint(0, len(self.positions) - 1)
            spark = copy.deepcopy(self.positions[rnd])

            # 随机维度进行变异
            rnd_dim_num = round(self.dims * np.random.random())
            rnd_dim = random.sample(range(self.dims), rnd_dim_num)
            g = random.gauss(1, 1)
            for j in rnd_dim:
                spark.position[j] *= g
                self.mapping(spark, j)
            spark.cal_fitness()
            new_positions.append(spark)
        self.positions += new_positions

    def mapping(self, spark, j):
        if spark.position[j] < self.realbounds[j][0] or spark.position[j] > self.realbounds[j][1]:
            spark.position[j] = self.realbounds[j][0] + \
                                abs(spark.position[j]) % (self.realbounds[j][1] - self.realbounds[j][0])

    def selection(self):
        new_positions = []
        new_positions.append(self.best)
        n_pop = len(self.positions)

        for i in range(n_pop):
            dist = sum([self.positions[i].cal_dist(self.positions[j]) for j in range(n_pop)])
            self.positions[i].Ri = dist
            
        sum_Ri = sum([self.positions[i].Ri for i in range(n_pop)])
        px = np.array([self.positions[i].Ri / sum_Ri for i in range(n_pop)]).cumsum()

        for i in range(self.n_pop):
            rnd = np.random.uniform()
            index = 0
            for j in range(n_pop - 1):
                if j == 0 and rnd < px[j]:
                    index = j
                elif rnd >= px[j] and rnd < px[j + 1]:
                    index = j + 1
            new_positions.append(self.positions[index])

        self.positions = new_positions

    def run(self):
        self.init_pop()
        self.cost = []
        count = 0
        while count < self.n_iter:
            self.explosion()
            self.mutation()
            self.update_fitness()
            self.best = copy.deepcopy(self.positions[self.best_ind])
            self.selection()
            self.cost.append(self.best.fitness)
            if self.best.fitness < self.epsilon:
                break
            count += 1


if __name__ == '__main__':
    realbounds = [[-100., 100.]] * 5
    initbounds = [[30., 50.]] * 5

    fwa = FWA(n_pop=15, n_iter=20, initbounds=initbounds, realbounds=realbounds, a=0.04, b=0.8, A_hat=40, m=50,
              n_mut=5)
    fwa.run()

# import matplotlib.pyplot as plt
#
# plt.plot(fwa.cost)
