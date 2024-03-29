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
    " For students they get used to keeping cellphone at hand",
    "but the overuse of it can be blamed to their parents",
    "Some parents don't set good examples for their children",
    "When they are together",
    "they just leave cellphones to kids and let them kill the time",
    "If they show the beautiful scenery around and teach kids to appreciate the world",
    "Then cellphone won't take up their time"
]


def get_embedding(sentences):
    # 从预训练模型中获取想要单词的embedding矩阵
    word_list = " ".join(sentences).split()
    word_list = [word.lower() for word in word_list]
    word_list = list(set(word_list))

    print(len(word_list))
    # 这个是预训练模型的文件
    filename = 'F:\\学习\\nlp\\glove-embedding\\glove.6B.300d.txt'
    save_path = 'F:\\Github\\nlp-implementation\\NNLM\\embedding.txt'
    data = open(filename, 'r', encoding='utf-8')
    writeObject = open(save_path, 'w')
    embedding = []
    for single in data:
        temp = single.split()
        if temp[0] in word_list:
            embedding.append(single)
            writeObject.write(single)
    writeObject.close()
    return embedding


embedding = get_embedding(sentences)
print(len(embedding))
# for i in embedding:
#     print(i)
