from src.cs import CS
import random
from src.greeting_utils import Greeting

if __name__ == '__main__':
    sents = ['mình muốn nhập vào sổ cái',
             'bạn tên là gì thế',
             'bạn xinh quá'
             ]
    cs = CS(threshold=0.45)
    greeting = Greeting()
    sentences, labels, map_greeting = greeting.sentences, greeting.labels, greeting.map_greeting
    cs.map_qa = map_greeting

    cs.fit(sentences, labels)
    pred = cs.predict(sents)
    print(sentences)
    print(labels)
    print(map_greeting)
    for pre in pred:
        res = random.choice(map_greeting[pre].split('|')).strip()
        print(res)
        print(map_greeting[pre])