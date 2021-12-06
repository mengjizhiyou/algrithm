import numpy as np


def objective(x):
    return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + (1 / 3) * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4


class MFA:
    def __init__(self, n_pop, bounds, n_iter, num_flames=1, b=1):
        self.n_pop = n_pop
        self.bounds = bounds
        self.n_iter = n_iter
        self.dim = len(bounds)
        self.num_flames = num_flames
        self.b = b
        self.a_linear_comp = -1
        self.best = None

    def init_moth_flame(self):
        lb = [l for l, h in self.bounds]
        ub = [h for l, h in self.bounds]
        positions = np.random.uniform(lb, ub, (self.n_pop, self.dim))
        fitness = np.apply_along_axis(objective, 1, positions)
        self.positions = np.hstack((positions, fitness.reshape(-1, 1)))
        self.flames = np.copy(self.positions[self.positions[:, -1].argsort()])

    def update_flame(self):
        population = np.vstack([self.flames, self.positions])
        self.flames = np.copy(population[population[:, -1].argsort()[:self.n_pop]])

    def update_moth(self):
        for i in range(self.n_pop):
            for j in range(self.dim):
                FD = abs(self.flames[i, j] - self.positions[i, j])
                rnd1 = np.random.random(1)
                rnd2 = (self.a_linear_comp - 1) * rnd1 + 1
                if i <= self.num_flames:
                    self.positions[i, j] = FD * np.exp(self.b * rnd2) * np.cos(rnd1 * 2 * np.pi) + self.flames[i, j]
                else:
                    self.positions[i, j] = np.clip(FD * np.exp(self.b * rnd2) * np.cos(rnd1 * 2 * np.pi)
                                                   + self.flames[self.num_flames, j],
                                                   self.bounds[j][0], self.bounds[j][1])
            self.positions[i, -1] = objective(self.positions[i, :-1])

    def run(self):
        count = 0
        self.cost = []
        self.init_moth_flame()
        self.best = np.copy(self.flames[0])
        while count < self.n_iter:
            self.num_flames = round(self.n_pop - count * ((self.n_pop - 1) / self.n_iter))
            self.a_linear_comp = -1 + count * (-1 / self.n_iter)
            self.update_moth()
            self.update_flame()
            count += 1
            if self.best[-1] > self.flames[0, -1]:
                self.best = np.copy(self.flames[0])
                self.cost.append(self.best[-1])


def plot_3D():
    x = np.arange(-5, 5, 0.01)
    y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(x, y)
    z = [objective(v) for v in np.vstack((X.flatten(), Y.flatten())).T]
    Z = np.array(z).reshape(X.shape)
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='rainbow', )
    min_ind = np.where(Z == Z.min())
    row, col = min_ind[0][0], min_ind[1][0]

    ax.scatter3D(X[row, col], Y[row, col], Z.min() - 600, color='r', s=20)
    ax.text(X[row, col] + 1, Y[row, col], Z.min() - 1200, f'f(x)={round(Z.min(), 2)}')


if __name__ == '__main__':
    bounds = [[-5, 5]] * 2
    mfa = MFA(n_pop=20, bounds=bounds, n_iter=100, num_flames=1, b=1)
    mfa.run()

    import matplotlib.pyplot as plt

    plt.plot(mfa.cost)
