import numpy as np

def objective(x):
    """objective/fitness function"""
    return x[0] ** 2.0 + x[1] ** 2.0


class GA:
    def __init__(self, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
        """
        bounds:每个输入变量的边界；
        n_bits:超参数，定义了单个候选解中的比特数;
        n_iter：超参数，枚举固定数量的算法迭代；
        n_pop：超参数，控制种群大小;
        r_cross：超参数，交叉率，决定是否执行交叉，不执行则将父代复制到下一代；
        r_mut：超参数，控制低概率反转位。
        """
        self.bounds = bounds
        self.n_bits = n_bits
        self.n_pop = n_pop
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.r_mut = r_mut

    def encode(self):
        """encode numbers to bitstring.
        初始化：创建随机位串的总体。
        可以使用布尔值 True 和 False，字符串值'0'和'1'，或整数值 0 和 1。
        这里使用整数值。
        """
        self.pop = [np.random.randint(0, 2, n_bits * len(bounds)).tolist() for _ in range(self.n_pop)]

    def decode(self, bitstring):
        """decode bitstring to numbers"""
        decoded = list()
        # 在给定两个输入变量的情况下，实际的位字符串将有(16 * 2)= 32位
        largest = len(self.bounds) ** n_bits
        for i in range(len(self.bounds)):
            start, end = i * n_bits, (i * n_bits) + n_bits  # 提取子串
            substring = bitstring[start:end]
            chars = ''.join([str(s) for s in substring])  # 生成二进制子串
            integer = int(chars, 2)  # 二进制位串转整数
            # 将整数缩放到所需的范围
            value = self.bounds[i][0] + (integer / largest) * (self.bounds[i][1] - self.bounds[i][0])
            decoded.append(value)
        return decoded

    def selection(self, fitness, k=2):
        """该函数获取种群并返回 1个选中的父母。
        k值表示从群体中选的个体数，固定为 2，用于父母选择"""
        selection_ix = np.random.randint(len(self.pop))
        for ix in np.random.randint(0, len(self.pop), k):
            if fitness[ix] < fitness[selection_ix]:
                selection_ix = ix
        return self.pop[selection_ix]

    def crossover(self, p1, p2):
        """根据交叉率生成后代"""
        # 将父代复制到下一代
        c1, c2 = p1.copy(), p2.copy()
        # 抽取[0,1]范围内的随机数来确定是否执行crossover
        if np.random.rand() < self.r_cross:
            # 选择一个有效的分割点
            pt = np.random.randint(1, len(p1) - 2)
            # 执行crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    def mutation(self, bitstring):
        for i in range(len(bitstring)):
            if np.random.rand() > self.r_mut:
                bitstring[i] = 1 - bitstring[i]  # 翻转位

    def run(self):
        self.encode()
        best, best_obj = 0, objective(self.decode(self.pop[0]))
        for gen in range(self.n_iter):
            # 种群解码
            decoded = [self.decode(p) for p in self.pop]
            # 评估所有候选解
            fitness = [objective(d) for d in decoded]
            for i in range(self.n_pop):
                if fitness[i] < best_obj:
                    best, best_obj = self.pop[i], fitness[i]
            # 调用selection函数n_pop次，以创建父母列表
            selected = [self.selection(fitness) for _ in range(n_pop)]
            # 循环遍历父级列表，并创建一个用作下一代的子级列表，根据需要调用交叉和变异函数。
            children = list()
            for i in range(0, n_pop, 2):
                # 选择成对的父母
                p1, p2 = selected[i], selected[i + 1]
                for c in self.crossover(p1, p2):
                    self.mutation(c)
                    children.append(c)
            self.pop = children  # 替换种群
        return [best, best_obj]


bounds = [[-5.0, 5.0], [-5.0, 5.0]]  # 输入范围
n_iter = 100  # 总迭代数
n_bits = 16  # 每个输入变量的比特数，并将其设置为16位
n_pop = 100  # 群体大小
r_cross = 0.9  # 交叉率
r_mut = 1.0 / (float(n_bits) * len(bounds))  # 突变率
ga = GA(bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
best, best_obj = ga.run()
print('Done!')
decoded = ga.decode(best)
print('objective(%s) = %f' % (decoded, best_obj))
