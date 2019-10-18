import numpy as np

N_STEP = 3
N_HIDDEN = 2
N_DIMENSION = 50
N_CLASS = 127
lr = 0.01
# 规定一下，输入X是列向量，格式为(N_STEP*N_DIMENSION, 1)

# embedding_path = 'embedding.txt'
# embedding_raw = open(embedding_path, 'r', encoding='utf-8')
#
#
# embedding_list = []
# for single in embedding_raw:
#     single = single.split()
#     word_list.append(single[0])
#     temp = [float(weight) for weight in single[1:]]
#     embedding_list.append(temp)

# 这里是把embedding的问题解决了, 从大的预训练集中抽取这个例子中要的单词的embedding先用着
# embedding = np.array(embedding_list)

# 使用随机初始化的embedding
embedding = np.random.random((N_CLASS, N_DIMENSION))

# sentences = ["i like dog", "i love coffee", "i hate milk"]
sentences = [
    'My father is a typical man',
    'He is not very talkative',
    'When other fathers say how much they love their children',
    'my father just keep quiet',
    'He barely says sweet words to me',
    'But he will never miss every important moment for me',
    'He is always one of the audience and watch my performance',
    'My father tells me to study with passion and he sets the good example for me',
    'because he loves his work',
    'Sometimes he shows me his design of work',
    'and I admire him',
    'I know my father loves me so much',
    "Though he doesn't talk much",
    "he will be right by my side whenever I need him",
    "When cellphone became popular many years ago",
    "parents and school at first banned students to take cellphone into classroom",
    "even the small child has owned a cellphone",
    "The use of cellphone can't be banned",
    "For students they get used to keeping cellphone at hand",
    "but the overuse of it can be blamed to their parents",
    "Some parents don't set good examples for their children",
    "When they are together",
    "they just leave cellphones to kids and let them kill the time",
    "If they show the beautiful scenery around and teach kids to appreciate the world",
    "Then cellphone won't take up their time"
]

word_list = []
for sentence in sentences:
    temp = sentence.split()
    for single in temp:
        word_list.append(single.lower())
word_list = list(set(word_list))

print(len(word_list))

word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
N_CLASS = len(word_dict)
print(word_list)


def get_number(words):
    # 给定单词的列表，返回对应单词编号的列表
    numbers = []
    for word in words:
        numbers.append(word_dict[word.lower()])
    return numbers


def make_batch(sentences):
    input_raw = []
    target_raw = []
    data_raw = []
    # for sen in sentences:
    #     word = sen.split()
    #     input = [word_dict[n] for n in word[:-1]]
    #     target = word_dict[word[-1]]
    #
    #     input_raw.append(input)
    #     target_raw.append(target)
    for sentence in sentences:
        data_raw.append(sentence.split())

    for data in data_raw:
        for i in range(len(data) - 3):
            segment = data[i: i + 3]
            next = data[i + 3]
            input_raw.append(get_number(segment))
            target_raw.append(word_dict[next.lower()])

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
# print(input_batch.shape)
# 升维去符合网络里面的计算要求
input_batch = input_batch[:, :, np.newaxis]
target_batch = target_batch[:, :, np.newaxis]

for epoch in range(1000):
    i = 0
    for single in input_batch:
        predict = model.forward(single)
        target = target_batch[i]
        model.backward(target, lr)
        i += 1
    if (epoch + 1) % 5 == 0:
        print('Epoch: ', '%04d' % (epoch + 1), ', Loss: ', model.loss[0][0])

test_data = ['but he will never']
while 1:
    words = input("请输入句子: ")
    test_data[0] = words
    test, target = make_batch(test_data)
    test = test[:, :, np.newaxis]
    test = test[0]
    predict = model.forward(test)
    index = 0
    for i in range(N_CLASS):
        if predict[i][0] > 0.1:
            index = i
            break
    predict_word = number_dict[index]
    print("predict word: ", predict_word)



