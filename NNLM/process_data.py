sentences = ["i like dog", "i love coffee", "i hate milk"]


def get_embedding(sentences):
    # 从预训练模型中获取想要单词的embedding矩阵
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))

    # 这个是预训练模型的文件
    filename = 'F:\\学习\\nlp\\glove-embedding\\glove.6B.100d.txt'
    data = open(filename, 'r', encoding='utf-8')
    embedding = []
    for single in data:
        temp = single.split()
        if temp[0] in word_list:
            embedding.append(single)
    return embedding


embedding = get_embedding(sentences)
for i in embedding:
    print(i)
