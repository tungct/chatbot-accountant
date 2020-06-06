import setting

class Greeting():
    def __init__(self):
        self.sentences, self.labels, self.map_greeting = self.load_data()

    def load_data(self):
        sentences, labels = [], []
        map_greeting = {
            0: 'Tôi sẽ trả lời sau'
        }
        with open(setting.GREETING_PATH, 'r', encoding='utf8', errors='ignore') as f:
            datas = f.readlines()
        all = [data.strip() for data in datas if data != '\n']
        len_greet_ = 0
        for i in range(len(all)):
            if i%2 == 0:
                sentence = all[i]
                sent_list = [sent.strip() for sent in sentence.split('|')]
                len_greet_ = len(sent_list)
                sentences += sent_list
            else:
                label_id, label = - (i//2 + 1), all[i]
                labels += [label_id] * len_greet_
                map_greeting[label_id] = all[i]
        return sentences, labels, map_greeting


if __name__ == '__main__':
    greeting = Greeting()
    print(greeting.load_data())
