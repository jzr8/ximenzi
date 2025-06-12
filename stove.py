# 炉子类  一个炉子实体化一个对象
class Stove:
    def __init__(self, date, number, pre, real):
        self.date = date
        self.number = number
        self.grade = [pre['grade'], real['grade']]
        # [符合flag，预计值，实测值，偏差绝对值， 偏差（预计-实测），预测的偏差， 动态权重偏差]
        self.Fe = [0, pre['Fe'], real['Fe'], 0, 0, 0, 0]
        self.Cl = [0, pre['Cl'], real['Cl'], 0, 0, 0, 0]
        self.C = [0, pre['C'], real['C'], 0, 0, 0, 0]
        self.N = [0, pre['N'], real['N'], 0, 0, 0, 0]
        self.O = [0, pre['O'], real['O'], 0, 0, 0, 0]
        self.Ni = [0, pre['Ni'], real['Ni'], 0, 0, 0, 0]
        self.Cr = [0, pre['Cr'], real['Cr'], 0, 0, 0, 0]
        self.hard = [0, pre['hard'], real['hard'], 0, 0, 0, 0]
        self.calculate_error()

    # 计算偏差
    def calculate_error(self):
        self.Fe[3] = (abs(self.Fe[1] - self.Fe[2]))
        self.Cl[3] = (abs(self.Cl[1] - self.Cl[2]))
        self.C[3] = (abs(self.C[1] - self.C[2]))
        self.N[3] = (abs(self.N[1] - self.N[2]))
        self.O[3] = (abs(self.O[1] - self.O[2]))
        self.Ni[3] = (abs(self.Ni[1] - self.Ni[2]))
        self.Cr[3] = (abs(self.Cr[1] - self.Cr[2]))
        self.hard[3] = (abs(self.hard[1] - self.hard[2]))

        self.Fe[4] = self.Fe[1] - self.Fe[2]
        self.Cl[4] = self.Cl[1] - self.Cl[2]
        self.C[4] = self.C[1] - self.C[2]
        self.N[4] = self.N[1] - self.N[2]
        self.O[4] = self.O[1] - self.O[2]
        self.Ni[4] = self.Ni[1] - self.Ni[2]
        self.Cr[4] = self.Cr[1] - self.Cr[2]
        self.hard[4] = self.hard[1] - self.hard[2]

    # 输入偏差指标 对各个指标 进行偏差判断
    def judge_error(self, stand):
        self.Fe[0] = 1 if (self.Fe[3] <= stand['Fe']) else -1
        self.Cl[0] = 1 if (self.Cl[3] <= stand['Cl']) else -1
        self.C[0] = 1 if (self.C[3] <= stand['C']) else -1
        self.N[0] = 1 if (self.N[3] <= stand['N']) else -1
        self.O[0] = 1 if (self.O[3] <= stand['O']) else -1
        self.Ni[0] = 1 if (self.Ni[3] <= stand['Ni']) else -1
        self.Cr[0] = 1 if (self.Cr[3] <= stand['Cr']) else -1
        self.hard[0] = 1 if (self.hard[3] <= stand['hard']) else -1

    # 输入偏差指标 对各个指标 进行偏差判断
    def judge_error_correct(self, stand):
        self.Fe[0] = 1 if (abs(self.Fe[4] - self.Fe[5]) <= stand['Fe']) else -1
        self.Cl[0] = 1 if (abs(self.Cl[4] - self.Cl[5]) <= stand['Cl']) else -1
        self.C[0] = 1 if (abs(self.C[4] - self.C[5]) <= stand['C']) else -1
        self.N[0] = 1 if (abs(self.N[4] - self.N[5]) <= stand['N']) else -1
        self.O[0] = 1 if (abs(self.O[4] - self.O[5]) <= stand['O']) else -1
        self.Ni[0] = 1 if (abs(self.Ni[4] - self.Ni[5]) <= stand['Ni']) else -1
        self.Cr[0] = 1 if (abs(self.Cr[4] - self.Cr[5]) <= stand['Cr']) else -1
        self.hard[0] = 1 if (abs(self.hard[4] - self.hard[5]) <= stand['hard']) else -1

    # 将日期转换为可以比较的整数
    def parse_date(self):
        month, day = map(int, self.date.split('_'))
        return month, day