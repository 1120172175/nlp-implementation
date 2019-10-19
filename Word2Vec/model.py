import numpy as np


N_CLASS = 14
N_HIDDEN = 300


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Word2Vec():
    def __init__(self):
        self.x = np.random.random((1, N_CLASS))
        self.embedding = np.random.random((N_CLASS, N_HIDDEN))
        self.w = np.random.random((N_CLASS, N_HIDDEN))
        self.y = np.random.random((N_CLASS, 1))
        self.S = np.random.random((N_CLASS, 1))
        self.D = np.random.random((N_CLASS, 1))
        self.loss = np.zeros((1, 1))

    def cal_loss(self, target, predict):
        predict = np.log(predict)
        return -np.dot(target.T, predict)

    def forward(self, x):
        self.x = x
        self.y = np.dot(self.w, (np.dot(self.x, self.embedding)).T)
        predict = softmax(self.y)
        self.S = predict
        return predict

    def backward(self, output, lr):
        self.loss = self.cal_loss(output, self.S)
        self.D = self.S - output
        dLoss_w = np.dot(self.D, np.dot(self.x, self.embedding))
        dLoss_emb = np.dot(self.D, np.dot(self.x, self.w))

        self.w -= lr * dLoss_w
        self.embedding -= lr * dLoss_emb


