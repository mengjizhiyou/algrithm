import numpy as np
from scipy.special import gamma


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class DA:
    def __init__(self, n_pop, bounds, n_iter, beta=1.5):
        self.n_pop = n_pop
        self.bounds = np.array(bounds)
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.beta = beta
        self.best = None
        self.worst = None
        self.velocity_max = np.array([(v[1] - v[0]) / 10 for v in self.bounds]).reshape(1, -1)

    def init_pop(self, size):
        lb = [l for l, h in self.bounds]
        ub = [h for l, h in self.bounds]
        pop = np.random.uniform(lb, ub, (size, self.dim))
        return pop

    def update_food_enemy(self):
        fitness = np.apply_along_axis(objective, 1, self.positions)
        self.food_pos = self.positions[np.argmin(fitness)]
        self.enemy_pos = self.positions[np.argmax(fitness)]

    def _cal_dist(self, type='p'):
        """找到半径内的邻居"""
        if type == 'p':
            dist = np.abs(self.positions - self.positions[np.newaxis,])
            return np.all(dist < self.radius, 2) - np.eye(self.n_pop)  # (n_pop,n_pop)
        elif type == 'f':
            dist = np.abs(self.positions - self.food_pos)
            return np.all(dist < self.radius, 1)  # (n_pop,)
        elif type == 'e':
            dist = np.abs(self.positions - self.enemy_pos)
            return np.all(dist < self.radius, 1)  # (n_pop,)

    def _cal_by_neighbours(self, matrix, values):
        """若有邻居，则求matrix/邻居个数，否则不变"""
        nc = np.repeat(self.nc, self.dim).reshape(self.n_pop, self.dim)
        # nc = self.nc.reshape(-1,1)
        ne = np.where(nc > 0)
        non_ne = np.where(nc == 0)
        matrix[ne] /= nc[ne]
        matrix[non_ne] = values[non_ne]
        return matrix

    def cal_actions(self):
        n_matrix = self._cal_dist(type='p').reshape((self.n_pop, self.n_pop, 1))
        n_food = self._cal_dist(type='f').reshape(-1, 1)
        n_enemy = self._cal_dist(type='e').reshape(-1, 1)

        p_matrix = self.positions[np.newaxis,].repeat(self.n_pop, axis=0)  # (n_pop, n_pop, dim)
        v_matrix = self.velocity[np.newaxis,].repeat(self.n_pop, axis=0)  # (n_pop, n_pop, dim)

        self.nc = np.sum(n_matrix, axis=1).reshape(self.n_pop, )  # neighbour_count
        self.non_ne = np.where(self.nc == 0)  # non neighbour index
        self.ne = np.where(self.nc > 0)  # neighbour index

        self.S = -np.sum((self.positions - p_matrix) * n_matrix, 1)

        self.A = np.sum(v_matrix * n_matrix, 1)
        self.A = self._cal_by_neighbours(self.A, self.velocity)

        self.C = np.sum(p_matrix * n_matrix, 1)
        self.C = self._cal_by_neighbours(self.C, self.positions) - self.positions

        self.F = n_food * (self.food_pos - self.positions)
        self.E = n_enemy * (self.enemy_pos + self.positions)

    def update_delta(self):
        # Update velocity
        self.velocity = self.velocity * self.w + self.S * self.s + \
                        self.A * self.a + self.C * self.c + \
                        self.F * self.f + self.E * self.e
        self.velocity = np.clip(self.velocity, np.zeros(self.velocity_max.shape), self.velocity_max)

    def update_position(self):
        # 根据邻居更新位置
        self.positions[self.ne] += self.velocity[self.ne]

        # 随机游走方式更新位置
        self.levy_flight()
        self.positions[self.non_ne] += self.positions[self.non_ne] * self.levy

        # 边界检查
        self.velocity[np.where(self.positions < self.bounds[:, 0])] *= -1
        self.velocity[np.where(self.positions > self.bounds[:, 1])] *= -1
        self.positions = np.clip(self.positions, self.bounds[:, 0], self.bounds[:, 1])

    def levy_flight(self):
        r1, r2 = np.random.rand(2)
        sig_numerator = gamma(1 + self.beta) * np.sin((np.pi * self.beta) / 2.0)
        sig_denominator = gamma((1 + self.beta) / 2) * self.beta * np.power(2, (self.beta - 1) / 2)
        sigma = np.power(sig_numerator / sig_denominator, 1 / self.beta)
        self.levy = (0.01 * r1 * sigma) / (abs(r2) ** (1 / self.beta))

    def adaptive_param(self, count):
        self.w = 0.9 - count * (0.5 / self.n_iter)
        self.const = 0.1 - count * (0.1 / (self.n_iter / 2))
        self.const = 0.0 if self.const < 0.0 else self.const
        r1, r2, r3, r4 = np.random.random(4)
        self.s = 2 * r1 * self.const  # Seperation Weight
        self.a = 2 * r2 * self.const  # A Weight
        self.c = 2 * r3 * self.const  # C Weight
        self.f = 2 * r4  # Food Attraction Weight
        self.e = 1 * self.const  # E distraction Weight

    def update_radius(self, count):
        self.radius = (self.bounds[:, 1] - self.bounds[:, 0]) * (0.25 + count / self.n_iter * 2)

    def run(self):
        self.positions = self.init_pop(self.n_pop)
        self.velocity = self.init_pop(self.n_pop)  # delta
        self.update_food_enemy()
        self.best = np.copy(self.food_pos)
        self.worst = np.copy(self.enemy_pos)

        count = 0
        self.cost = []
        while count < self.n_iter:
            self.update_radius(count)
            self.cal_actions()
            self.adaptive_param(count)
            self.update_delta()
            self.update_position()
            self.update_food_enemy()

            if objective(self.best) > objective(self.food_pos):
                self.best = np.copy(self.food_pos)
                self.cost.append(objective(self.best))
            else:
                self.food_pos = np.copy(self.best)

            if objective(self.worst) < objective(self.enemy_pos):
                self.worst = np.copy(self.enemy_pos)
            else:
                self.enemy_pos = np.copy(self.worst)

            count += 1


if __name__ == '__main__':
    da = DA(n_pop=5, bounds=[[-5, 5]] * 2, n_iter=300)
    da.run()

    import matplotlib.pyplot as plt

    plt.plot(da.cost)
