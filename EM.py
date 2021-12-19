import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
np.random.seed(30)


def objective(old_mu, new_mu):
    """
    检查参数是否收敛；
    收敛指标：参数的距离
    """
    return np.sqrt(np.sum(np.square(np.array(old_mu) - np.array(new_mu))))


class EM:
    """估计两个高斯分布的参数"""

    def __init__(self, num, mu1, sig1, mu2, sig2, n_iter=10, epsilon=1e-2):
        self.num = num
        self.mu1 = mu1
        self.sig1 = sig1
        self.mu2 = mu2
        self.sig2 = sig2
        self.n_iter = n_iter
        self.epsilon = epsilon
        x1, y1 = self.init_gaussian(mu1, sig1)
        x2, y2 = self.init_gaussian(mu2, sig2)
        self.x = np.concatenate((x1, x2))
        self.y = np.concatenate((y1, y2))
        self.labels = [1] * self.num + [2] * self.num
        self.init_param()

    def init_gaussian(self, mu, sig):
        x, y = np.random.multivariate_normal(mu, sig, self.num).T
        return x, y

    def init_param(self):
        self.mu1_guess = [1, 1]
        self.sig1_guess = [[1, 0], [0, 1]]
        self.mu2_guess = [4, 4]
        self.sig2_guess = [[1, 0], [0, 1]]
        self.lambda_ = [0.4, 0.6]
        self.labels_guess = np.random.choice(2, self.num * 2) + 1

    def plot_label(self, x, y, labels):
        plt.figure()
        plt.scatter(x, y, 24, c=labels)

    def plot_hist(self, val):
        plt.figure()
        plt.hist(val)

    def cal_prob(self, val, mu, sig, lambda_):
        """计算来自给定高斯分布的概率"""
        p = np.apply_along_axis(norm.pdf, 0, val, mu, np.diag(sig))
        return p.cumprod()[-1] * lambda_

    def E_step(self):
        """计算来自每个高斯的概率并指定类别"""
        p_cluster1 = [*map(lambda val: self.cal_prob(val, self.mu1_guess, self.sig1_guess, self.lambda_[0]),
                           np.vstack((self.x, self.y)).T)]
        p_cluster2 = [*map(lambda val: self.cal_prob(val, self.mu2_guess, self.sig2_guess, self.lambda_[1]),
                           np.vstack((self.x, self.y)).T)]
        self.labels_guess = [*map(lambda x, y: (x < y) + 1, p_cluster1, p_cluster2)]

    def M_step(self):
        """找到最可能的参数值"""
        cluster1 = np.argwhere(np.array(self.labels_guess) == 1).flatten()
        cluster2 = np.argwhere(np.array(self.labels_guess) == 2).flatten()
        prob_c1 = len(cluster1) / self.num / 2.
        prob_c2 = 1-prob_c1
        self.lambda_ = [prob_c1, prob_c2]
        self.mu1_guess = [self.x[cluster1].mean(), self.y[cluster1].mean()]
        self.sig1_guess = [[self.x[cluster1].std(), 0], [0, self.y[cluster1].std()]]
        self.mu2_guess = [self.x[cluster2].mean(), self.y[cluster2].mean()]
        self.sig2_guess = [[self.x[cluster2].std(), 0], [0, self.y[cluster2].std()]]

    def run(self):
        self.cost = []
        dist = objective(self.mu1 + self.mu2, self.mu1_guess + self.mu2_guess)
        count = 0
        while dist > self.epsilon and count<self.n_iter:
            self.E_step()
            self.M_step()
            dist = objective(self.mu1 + self.mu2, self.mu1_guess + self.mu2_guess)
            self.cost.append(dist)
            count+=1


if __name__ == '__main__':
    mu1 = [0, 5]
    sig1 = [[2, 0], [0, 3]]
    mu2 = [5, 0]
    sig2 = [[4, 0], [0, 1]]
    em = EM(num=100, mu1=mu1, sig1=sig1, mu2=mu2, sig2=sig2, n_iter=10)
    em.run()
    plt.plot(em.cost)
    em.plot_label(em.x,em.y,em.labels)
    em.plot_label(em.x,em.y,em.labels_guess)
    em.plot_hist(em.x)
    em.plot_hist(em.y)
