import numpy as np
from model import *
embed_size = 1
hidden_size = 64
n_class = 1
window_size = 10

start = 0
end = 2 * np.pi
length = end - start

steps = 100
input_data = []
output_data = []
for i in range(steps - window_size):
    now_start = start + i * length / steps
    temp = []
    for j in range(window_size):
        x = now_start + j * length / steps
        temp.append(np.sin(x))
    input_data.append(temp)
    output_data.append(np.cos(now_start + window_size * length / steps))
input_data = np.array(input_data)
print(input_data.shape)
output_data = np.array(output_data)

model = RNN(window_size, hidden_size, embed_size, n_class)
# print(output_data)
EPOCH = 5000
# for i in range(EPOCH):
#     data_in = input_data[i]
