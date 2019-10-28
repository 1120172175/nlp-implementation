import numpy as np


def diag(input_matrix):
    # This function will transform h*1 matrix into h*h matrix
    matrix_size = len(input_matrix)
    output_matrix = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        output_matrix[i][i] = input_matrix[i][0]
    return output_matrix


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class Node:
    def __init__(self):
        self.h = 0
        self.d_tanh = 0
        self.o = 0
        self.e = 0


class RNN:
    def __init__(self, window_size, hidden_size, embed_size, n_class):
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.n_classs = n_class
        self.embed_size = embed_size

        self.w = np.random.random((hidden_size, hidden_size))
        self.u = np.random.random((hidden_size, embed_size))
        self.v = np.random.random((n_class, hidden_size))
        self.b = np.random.random((hidden_size, 1))
        self.c = np.random.random((n_class, 1))
        self.x = []

        self.hidden_node_list = []
        # window_size means how many words in each input batch
        # for example, if i input "I love apple" into the network, then the window_size is 3
        for i in range(window_size):
            # initialize all the nodes in the hidden layer
            # then the net will calculate some related value in the forward part
            new_node = np.random.random((hidden_size, 1))
            self.hidden_node_list.append(new_node)

    def forward(self, x):
        self.x = x

        # default h0 = 0
        h0 = np.zeros((self.hidden_size, 1))

        # before calculate, we need to clear the node list of the net
        self.hidden_node_list.clear()

        # calculate some intermediate variables
        for i in range(self.window_size):
            if i == 0:
                new_node = Node()
                new_node.h = np.tanh(np.dot(self.u, x[0]) + np.dot(self.w, h0) + self.b)
                new_node.d_tanh = 1 - new_node.h ** 2
                new_node.o = np.dot(self.v, new_node.h) + self.c
                self.hidden_node_list.append(new_node)


        # suppose we only need the last output state
        # return h(t), which is the last one in h list
        return self.hidden_node_list[-1]

    def backward(self, target_output, lr):
        h0 = np.zeros((self.hidden_size, 1))

        # then calculate the gradient in each hidden layer
        predict = softmax(self.hidden_node_list[-1].o)
        for i in range(self.window_size):
            j = self.window_size - i - 1
            if i == 0:
                self.hidden_node_list[j].e = np.dot(self.v.T, predict - target_output)
            else:
                temp = np.dot(self.w.T, diag(self.hidden_node_list[j + 1].d_tanh))
                self.hidden_node_list[j].e = np.dot(temp, self.hidden_node_list[j + 1].e)
        dloss_w = np.zeros((self.hidden_size, self.hidden_size))
        dloss_u = np.zeros((self.hidden_size, self.embed_size))
        dloss_b = np.zeros((self.hidden_size, 1))

        dloss_c = predict - target_output
        dloss_v = np.dot(predict-target_output, self.hidden_node_list[-1].h.T)

        for i in range(self.window_size):
            temp = np.dot(diag(self.hidden_node_list[i].d_tanh), self.hidden_node_list[i].e)
            if i == 0:
                # seem unnecessary~
                dloss_w += 0
            else:
                dloss_w += np.dot(temp, self.hidden_node_list[i - 1].h.T)
            dloss_b += temp
            dloss_u += np.dot(temp, self.x[i].T)

        self.w -= lr * dloss_w
        self.u -= lr * dloss_u
        self.v -= lr * dloss_v
        self.b -= lr * dloss_b
        self.c -= lr * dloss_c


# test = np.random.random((10, 1))
# len = len(test)
# print(len)
