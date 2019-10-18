import numpy as np
import matplotlib.pyplot as plt
from process_data import make_batch
from model import Word2Vec

window_size = 1
N_CLASS = 1
N_HIDDEN = 300
lr = 0.01

sentences = [
    "i like dog", "i like cat", "i like animal",
    "dog cat animal", "apple cat dog like", "dog fish milk like",
    "dog cat eyes like", "i like apple", "apple i hate",
    "apple i movie book music like", "cat dog hate", "cat dog like"
]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))

input_batch, output_batch = make_batch(sentences)

model = Word2Vec()

for epoch in range(30000):
    for i in range(len(input_batch)):
        model.forward(input_batch[i])
        model.backward(output_batch[i], 0.01)
    if (epoch + 1) % 500 == 0:
        print('Epoch: ', '%04d' % (epoch + 1), ', Loss: ', model.loss[0][0])

for i, label in enumerate(word_list):
    W = model.w
    WT = model.w.T
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()