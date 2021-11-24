import numpy as np
import matplotlib.pyplot as plt


def objective(x):
    """目标函数"""
    return x[0] ** 2.0


class SA:
    def __init__(self, X, bounds, n_iter, sigma, T):
        self.X = X
        self.bounds = bounds
        self.best = X[0]  # 初始化
        self.best_obj = objective([self.best])
        self.n_iter = n_iter
        self.sigma = sigma
        self.T = T
        self.cur, self.cur_obj = self.best, self.best_obj
        self.dims = len(bounds)

    def run(self):
        self.cost = []
        for i in range(self.n_iter):

            # 假设候选解来自正态分布 N(self.cur,sigma)
            candidate = self.cur + np.random.randn(self.dims) * self.sigma
            candidate_obj = objective(candidate)

            # 最优参数和目标值
            if candidate_obj < self.best_obj:
                self.cur, self.cur_obj = candidate, candidate_obj
                self.best, self.best_obj = candidate, candidate_obj
                self.cost.append(self.best_obj)

            # 是否接受更差的解
            diff = candidate_obj - self.best_obj
            T = self.T / float(i + 1)  # 当前迭代次数的温度
            metropolis = np.exp(-diff / T)
            if diff < 0 and np.random.rand() < metropolis:
                self.cur, self.cur_obj = candidate, candidate_obj


def plot_origin(X, objs):
    """不同X的目标值"""
    plt.subplot(131)
    plt.plot(X, objs)
    x_optima = 0.0
    plt.axvline(x=x_optima, ls='--', color='red')
    plt.xlabel('X')
    plt.ylabel('objective')


def plot_T(T, n_iter):
    """T随迭代次数的变化"""
    plt.subplot(132)
    temps = [T / float(i + 1) for i in range(n_iter)]
    plt.plot(range(n_iter), temps)
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')


def plot_metropolis(T, n_iter):
    temps = [T / float(i + 1) for i in range(n_iter)]
    # 坏解接受概率
    diff = [0.01, 0.1, 1.0]
    ax = plt.subplot(133)
    for d in diff:
        metropolis = [np.exp(-d / t) for t in temps]
        label = 'diff=%.2f' % d
        ax.plot(range(n_iter), metropolis, label=label)
        plt.xlabel('Iteration')
        plt.ylabel('Metropolis Criterion')
        plt.legend()
        plt.show()
    # 解决方案越差(差异越大)，模型就越不可能接受更差的解决方案


if __name__ == '__main__':
    bounds = [-5, 5]
    X = np.arange(bounds[0], bounds[1], 0.1)
    T = 10
    n_iter = 1000

    # -----------------------------------------------------
    sa = SA(X, bounds, n_iter, sigma=0.2, T=T)
    sa.run()
    plt.figure()
    plt.plot(sa.cost)  # cost变化情况

    # 目标函数的评估大约有35个变化，最初变化很大，
    # 随着算法收敛到最优值，在搜索接近尾声时变化很小，甚至难以察觉

    # -----------------------------------------------------
    objs = [objective([x]) for x in X]
    plt.figure()
    plot_origin(X, objs)
    plot_T(T, n_iter)
    plot_metropolis(T, n_iter)
