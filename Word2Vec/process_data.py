import numpy as np


sentences = [
    "i like dog", "i like cat", "i like animal",
    "dog cat animal", "apple cat dog like", "dog fish milk like",
    "dog cat eyes like", "i like apple", "apple i hate",
    "apple i movie book music like", "cat dog hate", "cat dog like"
]

# 表示取当前单词前后的一个单词作为输出
window_size = 1
N_CLASS = 1


def make_batch(sentences):
    input_batch = []
    output_batch = []

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    N_CLASS = len(word_list)
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}

    for sentence in sentences:
        temp = sentence.split()
        for i in range(len(temp)):
            for j in range(1, window_size + 1):
                if i - j >= 0:
                    input_batch.append(word_dict[temp[i]])
                    output_batch.append(word_dict[temp[i - j]])
                if i + j < len(temp):
                    input_batch.append(word_dict[temp[i]])
                    output_batch.append(word_dict[temp[i + j]])

    input = []
    output = []
    for i in range(len(input_batch)):
        temp_in = np.zeros((1, N_CLASS))
        temp_out = np.zeros((N_CLASS, 1))
        temp_in[0][input_batch[i]] = 1
        temp_out[output_batch[i]][0] = 1

        input.append(temp_in)
        output.append(temp_out)
    return input, output


