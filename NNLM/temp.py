import numpy as np
# import process_data

N_STEP = 2
N_HIDDEN = 2
N_DIMENSION = 50
N_CLASS = 7
lr = 0.01
# 规定一下，输入X是列向量，格式为(N_STEP*N_DIMENSION, 1)

embedding_path = 'embedding.txt'
embedding_raw = open(embedding_path, 'r', encoding='utf-8')

word_list = []
embedding_list = []
for single in embedding_raw:
    single = single.split()
    word_list.append(single[0])
    temp = [float(weight) for weight in single[1:]]
    embedding_list.append(temp)

# 这里是把embedding的问题解决了, 从大的预训练集中抽取这个例子中要的单词的embedding先用着
# embedding = np.array(embedding_list)
embedding = np.random.random((7, 50))
# print(embedding.shape)
sentences = ["i like dog", "i love coffee", "i hate milk"]


word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
N_CLASS = len(word_dict)


def make_batch(sentences):
    input_raw = []
    target_raw = []

    for sen in sentences:
        word = sen.split()
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]

        input_raw.append(input)
        target_raw.append(target)

    input_batch = []
    target_batch = []
    for single in input_raw:
        temp = []
        for i in single:
            temp = temp + list(embedding[i])
        input_batch.append(temp)
    input_batch = np.array(input_batch)

    for i in target_raw:
        temp = np.zeros(N_CLASS)
        temp[i] = 1
        target_batch.append(temp)
    target_batch = np.array(target_batch)

    return input_batch, target_batch


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


# 我靠 给整忘了，搞到最后发现embedding还没准备
# 好吧，不用准备embedding了，
class NNLM():
    def __init__(self):
        self.X = np.random.random((N_STEP * N_DIMENSION, 1))
        self.C = np.random.random((N_CLASS, N_DIMENSION))
        self.W = np.random.random((N_CLASS, N_STEP * N_DIMENSION))
        self.b = np.random.random((N_CLASS, 1))
        self.H = np.random.random((N_HIDDEN, N_STEP * N_DIMENSION))
        self.d = np.random.random((N_HIDDEN, 1))
        self.U = np.random.random((N_CLASS, N_HIDDEN))
        self.tanh = np.random.random((N_HIDDEN, 1))
        self.loss = np.random.random((1, 1))

        # y是神经元输出结果
        self.y = np.random.random((N_CLASS, 1))
        # S是softmax后的结果
        self.S = np.random.random((N_CLASS, 1))

    def cal_loss(self, target, predict):
        predict = np.log(predict)
        # print('predict:')
        # print(predict)
        return -np.dot(target.T, predict)

    # 网络的输入格式是(n-1)m*1的向量
    def forward(self, x):
        self.X = x
        output = np.dot(self.W, x) + self.b
        tanh = np.tanh(np.dot(self.H, x) + self.d)
        self.tanh = tanh
        output += np.dot(self.U, tanh)
        self.y = output
        output = softmax(output)
        self.S = output
        return output

    def backward(self, output, lr):
        # 先求softmax的导数矩阵D(n * n)的那个
        self.loss = self.cal_loss(output, self.S)

        dLoss_y = self.S - output
        dLoss_W = np.dot(dLoss_y, self.X.T)
        dLoss_b = dLoss_y
        dLoss_U = np.dot(dLoss_y, self.tanh.T)

        dLoss_H = np.multiply(np.dot(self.U.T, dLoss_y), d_tanh(np.dot(self.H, self.X) + self.d))

        # 在这里赋值是为了避免重复运算dLoss_d的值
        dLoss_d = dLoss_H

        dLoss_H = np.dot(dLoss_H, self.X.T)

        self.b -= lr * dLoss_b
        self.d -= lr * dLoss_d
        self.W -= lr * dLoss_W
        self.H -= lr * dLoss_H
        self.U -= lr * dLoss_U


model = NNLM()
input_batch, target_batch = make_batch(sentences)

# 升维去符合网络里面的计算要求
input_batch = input_batch[:, :, np.newaxis]
target_batch = target_batch[:, :, np.newaxis]
# for i in input_batch:
#     print(len(i))
# print(target_batch.shape)
# print(input_batch)

for epoch in range(1000):
    i = 0
    for input in input_batch:
        predict = model.forward(input)
        target = target_batch[i]
        model.backward(target, lr)
        i += 1
    if (epoch + 1) % 5 == 0:
        print('Epoch: ', '%04d' % (epoch + 1), ', Loss: ', model.loss[0][0])

index = 2
test_in = input_batch[index]
print(model.forward(test_in))
print(target_batch[index])

