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

# 还有就是数据处理的问题，需要将原始的sentences变成one-hot的形式

word_list = " ".join(sentences).split()
word_list = list(set(word_list))

word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}

data_raw = []
input = []
target = []


def get_number(words):
    # 给定单词的列表，返回对应单词编号的列表
    numbers = []
    for word in words:
        numbers.append(word_dict[word])
    return numbers



for sentence in sentences:
    data_raw.append(sentence.split())

for data in data_raw:
    for i in range(len(data) - 3):
        segment = data[i: i + 3]
        next = data[i + 3]
        input.append(get_number(segment))
        target.append(word_dict[next])



# print(input)
# for segment in input:
#     output = []
#     for number in segment:
#         output.append(number_dict[number])
#     print(output)
