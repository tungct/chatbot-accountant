from flask import *
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
sys.path.insert(0, '.')
import setting
from src.cs import CS
from src.greeting_utils import Greeting
import random

cs = CS()
X, y = cs.data_utils.sent_tokens, cs.data_utils.labels

cs_greeting = CS()
greeting = Greeting()
X_greeting, y_greeting, map_greeting = greeting.sentences, greeting.labels, greeting.map_greeting
cs_greeting.map_qa = map_greeting
# X, y = X + X_greeting, y + y_greeting

cs.fit(X, y)
cs_greeting.fit(X_greeting, y_greeting)

app = Flask(__name__, static_url_path="/static")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/m', methods=['POST'])
def response_message():
    question = request.form['m']
    cl_id = cs.predict([question])[0]
    print(cl_id)
    with open(setting.LOG_PATH, 'a') as f:
        f.write(question + '\n' + str(cl_id) + '\n' + '---' + '\n')
    if cl_id == 0:
        cl_other_id = cs_greeting.predict([question])[0]
        if cl_other_id != 0:
            answer = random.choice(map_greeting[cl_other_id].split('|')).strip()
        else:
            answer = cs.response_answer(0)
    else:
        answer = cs.response_answer(cl_id)
    return json.dumps({
        'message': answer
    })
if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run(host='0.0.0.0', port=9090)