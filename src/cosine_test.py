import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
sys.path.insert(0, '.')
from src.cs import CS
from src.greeting_utils import Greeting
import random

if __name__ == '__main__':
    cs = CS()
    X, y = cs.data_utils.sent_tokens, cs.data_utils.labels

    cs_greeting = CS()
    greeting = Greeting()
    X_greeting, y_greeting, map_greeting = greeting.sentences, greeting.labels, greeting.map_greeting
    cs_greeting.map_qa = map_greeting

    cs.fit(X, y)
    cs_greeting.fit(X_greeting, y_greeting)
    sentences = ['hihi']
    pred = cs.predict(sentences)
    cl_id = pred[0]
    if cl_id == 0:
        cl_other_id = cs_greeting.predict(sentences)[0]
        if cl_other_id != 0:
            answer = random.choice(map_greeting[cl_other_id].split('|')).strip()
        else:
            answer = cs.response_answer(-1)
    else:
        answer = cs.response_answer(cl_id)
    print(answer)
