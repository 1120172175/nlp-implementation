# label_data = data = [(0.05, 'c1'), (0.25, 'c2'), (0.03, 'c3'), (0.06, 'c4'), (0.10, 'c5'), (0.11, 'c6'), (0.36, 'c7'), (0.04, 'c8')]
# train_data = [
#     ('i love you', 'positive'),
#     ('he loves me', 'positive'),
#     ('she likes basketball', 'positive'),
#     ('i hate you', 'negative'),
#     ('sorry for that', 'negative'),
#     ('that is awful', 'negative')
# ]


def make_batch(data):
    word_list = [tuple[0] for tuple in data]

    word_list = " ".join(word_list).split()

    word_list = list(set(word_list))
    # print(word_list)
    word2number = {word: i for i, word in enumerate(word_list)}
    number2word = {i: word for i, word in enumerate(word_list)}
    input_batch = []
    target_batch = []
    label_data = {}
    for tuple in data:
        sentence = tuple[0].split()
        temp = [word2number[word] for word in sentence]
        input_batch.append(temp)
        # temp.clear()
        target_batch.append(tuple[1])
        if tuple[1] in label_data.keys():
            label_data[tuple[1]] += 1
        else:
            label_data[tuple[1]] = 1
    length = len(data)
    for label in label_data.keys():
        label_data[label] /= length
    label_data = [(number, label) for label, number in label_data.items()]
    return word_list, word2number, number2word, input_batch, target_batch, label_data


# word_list, word2number, number2word, input_batch, target_batch, label_data = make_batch(train_data)
# print(word_list)
# print(word2number)
# print(number2word)
# print(input_batch)
# print(target_batch)
# print(label_data)

