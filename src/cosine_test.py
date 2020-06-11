import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
sys.path.insert(0, '.')
from src.cs import CS
from src.greeting_utils import Greeting
import random

if __name__ == '__main__':
    cs = CS(threshold=0.45)
    X, y = cs.data_utils.sent_tokens, cs.data_utils.labels

    cs_greeting = CS(threshold=0.45)
    greeting = Greeting()
    X_greeting, y_greeting, map_greeting = greeting.sentences, greeting.labels, greeting.map_greeting
    cs_greeting.map_qa = map_greeting

    cs.fit(X, y)
    cs_greeting.fit(X_greeting, y_greeting)
    sentences = ['có gì vui']
    pred = cs.predict(sentences)
    cl_id = pred[0]
    print(cl_id)
    if cl_id == 0:
        cl_other_id = cs_greeting.predict(sentences)[0]
        print(cl_other_id)
        if cl_other_id != 0:
            answer = random.choice(map_greeting[cl_other_id].split('|')).strip()
        else:
            answer = cs.response_answer(-1)
    else:
        answer = cs.response_answer(cl_id)
    # print(answer)
