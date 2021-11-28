import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
import math


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class CSO:
    def __init__(self, X, n_iter, alpha, lambda_v, pa):
        self.X = X
        self.dim = X.shape[1]
        self.bound = [*zip(X.min(axis=0), X.max(axis=0))]
        self.n_pop = X.shape[0]
        self.n_iter = n_iter
        self.init_position()
        self.alpha = alpha
        self.lambda_v = lambda_v
        self.pa = pa
        self.best_nest = None

    def init_position(self):
        """初始化布谷鸟/鸟巢的位置"""
        self.position = np.zeros((self.n_pop, self.dim + 1))
        self.position[:, :-1] = self.X
        for i in range(self.n_pop):
            self.position[i, -1] = objective(self.position[i, :self.dim])

    def levy_flight(self):
        x1 = math.sin((self.lambda_v - 1.0) * (np.random.uniform(-0.5 * math.pi, 0.5 * math.pi))) / \
             (math.pow(math.cos((np.random.uniform(-0.5 * math.pi, 0.5 * math.pi))), (1.0 / (self.lambda_v - 1.0))))
        x2 = math.pow((math.cos((2.0 - self.lambda_v) * (np.random.uniform(-0.5 * math.pi, 0.5 * math.pi))) /
                       (-math.log(np.random.uniform(0.0, 1.0)))), ((2.0 - self.lambda_v) / (self.lambda_v - 1.0)))
        return x1 * x2

    def update_bird(self):
        """布谷鸟位置更新；"""
        random_bird = np.random.randint(self.n_pop, size=1)[0]
        new_solution = np.zeros((1, self.dim + 1))

        for j in range(self.dim):
            new_solution[0, j] = np.clip(
                self.position[random_bird, j] + self.alpha * self.levy_flight() * self.position[random_bird, j],
                self.bound[j][0], self.bound[j][1])

        new_solution[0, -1] = objective(new_solution[0, 0:self.dim])
        if self.position[random_bird, -1] > new_solution[0, -1]:
            self.position[random_bird, :] = np.copy(new_solution[0, :])

    def update_nest(self):
        """宿主构建新的鸟巢"""

        new_position = np.copy(self.position)

        # 被发现鸟巢的个数
        abandoned_nests = math.ceil(self.pa * self.n_pop) + 1

        # 鸟巢k根据鸟巢i更新鸟巢位置
        random_bird_j = np.random.randint(self.n_pop, size=1)[0]
        random_bird_k = np.random.randint(self.n_pop, size=1)[0]

        while random_bird_j == random_bird_k:
            random_bird_j = np.random.randint(self.n_pop, size=1)[0]

        # 自定义，根据目标值大小选取被抛弃的鸟巢
        nest_list = np.random.choice(range(self.n_pop), size=abandoned_nests,
                                     p=np.exp(self.position[:, -1]) / sum(np.exp(self.position[:, -1])))

        for i in range(0, self.n_pop):
            for j in range(0, len(nest_list)):
                discovery_rate = np.random.random(1)[0]
                if i == nest_list[j] and discovery_rate > self.pa:
                    # 更新鸟巢位置
                    for k in range(0, self.dim):
                        epsilon = np.random.random(1)[0]
                        new_position[i, k] = np.clip(new_position[i, k] + epsilon * (
                                new_position[random_bird_j, k] - new_position[random_bird_k, k]),
                                                     self.bound[k][0], self.bound[k][1])
            new_position[i, -1] = objective(new_position[i, :-1])
        self.position = new_position

    def run(self):
        """布谷鸟的位置=鸟巢的位置，最优布谷鸟位置=最优鸟巢位置"""
        count = 0
        self.cost = []
        self.best_nest = np.copy(self.position[self.position[:, -1].argsort()][0, :])
        while count <= self.n_iter:
            for i in range(0, self.n_pop):
                self.update_bird()
                self.update_nest()
            best = np.copy(self.position[self.position[:, -1].argsort()][0])

            if self.best_nest[-1] > best[-1]:
                self.best_nest = best
                self.cost.append(best[-1])
            count = count + 1


if __name__ == '__main__':
    X, _ = make_regression(n_samples=100, n_features=2)
    cso = CSO(X=X, n_iter=50, alpha=0.01, lambda_v=1.5, pa=0.25)
    cso.run()
    plt.plot(cso.cost)
    plt.xlabel('iterations')
    plt.ylabel('objective')
    # plt.legend()
