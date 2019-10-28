import numpy as np
from model import *
from process_data import *
embed_size = 10
lr = 0.01

# label_data = data = [(0.05, 'c1'), (0.25, 'c2'), (0.03, 'c3'), (0.06, 'c4'), (0.10, 'c5'), (0.11, 'c6'), (0.36, 'c7'), (0.04, 'c8')]
train_data = [
    ('i love you', 'positive'),
    ('he loves me', 'positive'),
    ('she likes basketball', 'positive'),
    ('i hate you', 'negtive'),
    ('sorry for that', 'negtive'),
    ('that is awful', 'negtive')
]

test_data = [
    ('he is awful', 'negative'),
    ('that is awful', 'negative'),
    ('i love that', 'positive'),
    ('she likes you', 'negative')
]


word_list, word2number, number2word, input_batch, target_batch, label_data = make_batch(train_data)

tree = make_huffman(label_data, embed_size)
print(tree)
label2code = {}
now_code = []
generate_huffman_code(tree, label2code, now_code)

code2label = {code: label for label, code in label2code.items()}

model = FastText(label_data, label2code, len(word_list), embed_size)

EPOCH = 3000
for epoch in range(EPOCH):
    loss = 0
    data_size = len(input_batch)
    for i in range(data_size):
        model.forward(input_batch[i])
        loss += model.cal_loss(target_batch[i])
        model.backward(target_batch[i], lr)
    # print('Epoch: ', '%04d' % (epoch), ', Loss: ', loss)
    if epoch % 500 == 0:
        print('Epoch: ', '%04d' % (epoch), ', Loss: ', loss)

# print(label2code)

for tuple in test_data:
    test = tuple[0]
    test = test.split()
    test = [word2number[word] for word in test]
    print(tuple[0], " :", model.forward(test))

# print(target_batch[0])

# for i in range(1000):
#     a = np.random.random((embed_size, 1))
#     b = 2 * np.random.random((embed_size, 1)) - 1
#     print(sigmoid(np.dot(a.T, b)[0][0]))
