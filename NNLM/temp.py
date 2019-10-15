import numpy as np
import matplotlib.pyplot as plt

INPUT_SIZE = 1
HIDDEN_SIZE = 32  # 隐藏层神经元个数
OUTPUT_SIZE = 1
lr = 0.05  # 学习率
TRAIN_DATA_SIZE = 2000  # 训练集数据样本个数
VALIDATE_DATA_SIZE = 2000  # 验证集数据样本个数
EPOCH = 2000  # 迭代次数


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))


class Net():
    def __init__(self):
        self.w1 = np.random.random((HIDDEN_SIZE, INPUT_SIZE))
        self.b1 = np.random.random((HIDDEN_SIZE, 1))

        self.w2 = np.random.random((OUTPUT_SIZE, HIDDEN_SIZE))
        self.b2 = np.random.random((OUTPUT_SIZE, 1))

        # 输入是列向量，input其实就是a0
        self.a0 = np.random.random((INPUT_SIZE, 1))
        self.z1 = np.random.random((HIDDEN_SIZE, 1))
        self.a1 = np.random.random((HIDDEN_SIZE, 1))
        self.z2 = np.random.random((OUTPUT_SIZE, 1))

    # 计算前向传播计算出来的结果
    def forward(self, input):
        self.a0 = input
        self.z1 = np.dot(self.w1, self.a0) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        return self.z2

    # 反向传播更新参数的值，更新w1 b1, w2 b2
    def backward(self, output, lr):
        dLoss_z2 = self.z2 - output
        dLoss_w2 = dLoss_z2 * self.a1.T
        dLoss_b2 = dLoss_z2

        dLoss_z1 = self.w2.T * dLoss_z2 * d_sigmoid(self.z1)
        dLoss_w1 = dLoss_z1 * self.a0.T
        dLoss_b1 = dLoss_z1

        self.w1 -= lr * dLoss_w1
        self.b1 -= lr * dLoss_b1

        self.w2 -= lr * dLoss_w2
        self.b2 -= lr * dLoss_b2


# 目标函数计算公式
def cal_fun(x):
    p1 = 0.4 * (x ** 2)
    # p2 = 0
    # p3 = 0
    p2 = 0.3 * np.sin(15 * x)
    p3 = 0.01 * np.cos(50 * x)
    y = p1 + p2 + p3
    return y


# 随机生成一个训练数据集
train_x = np.random.random((TRAIN_DATA_SIZE, INPUT_SIZE, 1))
train_y = cal_fun(train_x)


# 随机生成一个验证数据集
validate_x = np.random.random((VALIDATE_DATA_SIZE, INPUT_SIZE, 1))
validate_y = cal_fun(validate_x)

net = Net()  # 生成一个网络
# 迭代次数为EPOCH次
for epoch in range(EPOCH):
    print("epoch:", epoch)
    for i in range(TRAIN_DATA_SIZE):
        net.forward(train_x[i])  # 前向传播
        net.backward(train_y[i], lr)  # 反向传播并更新参数


# 预测结果存到prediction里面
prediction = np.random.random((VALIDATE_DATA_SIZE, INPUT_SIZE, 1))
for i in range(VALIDATE_DATA_SIZE):
    prediction[i] = net.forward(validate_x[i])  # 进行预测


x1 = np.random.rand(TRAIN_DATA_SIZE)
y1 = np.random.rand(TRAIN_DATA_SIZE)
y2 = np.random.rand(TRAIN_DATA_SIZE)


# 将原来三维的数据集变成一维的numpy数组用于画图
for i in range(VALIDATE_DATA_SIZE):
    x1[i] = validate_x[i][0][0]
    y1[i] = validate_y[i][0][0]
    y2[i] = prediction[i][0][0]


plt.plot(x1, y1, 'ro')  # 验证集上的数据图，红色
plt.plot(x1, y2, 'bo')  # 网络跑出来的数据图，蓝色
plt.draw()
plt.pause(0.5)