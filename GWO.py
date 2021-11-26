import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class GWO:
    def __init__(self, X, n_iter, linear_component=2):
        self.X = X
        self.dim = X.shape[1]
        self.bound = [*zip(X.min(axis=0), X.max(axis=0))]
        self.n_pop = X.shape[0]
        self.n_iter = n_iter
        self.linear_component = linear_component
        self.init_position()
        self.alpha, self.beta, self.delta = [self.best_position() for _ in range(3)]

    def init_position(self):
        self.position = np.zeros((self.n_pop, self.dim + 1))
        self.position[:, :-1] = self.X
        for i in range(self.n_pop):
            self.position[i, -1] = objective(self.position[i, :self.dim])

    def best_position(self):
        best = np.zeros((1, self.dim + 1))
        best[0, -1] = objective(best[0, :-1])
        return best

    def update_best(self):
        """根据目标值更新３个最优解决方案"""
        new_position = self.position.copy()
        for i in range(self.dim):
            best_positions = [self.alpha[0, -1], self.beta[0, -1], self.delta[0, -1]]
            ind = np.searchsorted(best_positions, new_position[i, -1])
            if ind < 3:
                [self.alpha, self.beta, self.delta][ind][0, :] = np.copy(self.position[i, :])

    def update_position(self):
        """根据目标值更新个体位置"""
        new_position = np.copy(self.position)
        for i in range(len(new_position)):
            for j in range(0, self.dim):
                x1 = self._encircling_prey(self.alpha[0, j], self.position[i, j])
                x2 = self._encircling_prey(self.beta[0, j], self.position[i, j])
                x3 = self._encircling_prey(self.delta[0, j], self.position[i, j])
                new_position[i, j] = self._hunting(j, x1, x2, x3)
            new_position[i, -1] = objective(new_position[i, :self.dim])
        self.position = new_position

    def _encircling_prey(self, best_position, position):
        """包围猎物"""
        r1, r2 = np.random.random(2)
        a = 2 * self.linear_component * r1 - self.linear_component
        c = 2 * r2
        d = abs(c * best_position - position)
        x = best_position - a * d
        return x

    def _hunting(self, j, x1, x2, x3):
        """狩猎"""
        new_pos = np.clip(np.mean([x1, x2, x3]), self.bound[j][0], self.bound[j][1])
        return new_pos

    def run(self):
        generation = 1
        self.cost = []
        while generation < self.n_iter:
            self.linear_component = 2 - generation * (2 / self.n_iter)
            self.update_best()
            self.update_position()
            self.cost.append([self.alpha[0, -1], self.beta[0, -1], self.delta[0, -1]])
            generation += 1


if __name__ == '__main__':
    X, _ = make_regression(n_samples=100, n_features=2)
    gwo = GWO(X=X, n_iter=100)
    cost = gwo.run()
    plt.plot(gwo.cost, label=['alpha', 'beta', 'delta'])
    plt.xlabel('iterations')
    plt.ylabel('objective')
    plt.legend()
