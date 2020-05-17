from flask import *
import setting
from src.clf_utils import Clf_Utils

clf_util = Clf_Utils()

app = Flask(__name__, static_url_path="/static")
@app.route('/')
def index():
    session['loc'] = []
    session['time'] = []
    session['weather'] = []
    session['bot_state'] = 1
    session['bot_msg'] = ""
    return render_template('index.html')

@app.route('/m', methods=['POST'])
def response_message():
    question = request.form['m']
    cl_id = clf_util.predict([question])
    with open(setting.LOG_PATH, 'a') as f:
        f.write(question + '\n' + str(cl_id[0]) + '\n' + '---' + '\n')
    answer = clf_util.response_answer(cl_id[0])
    return json.dumps({
        'message': answer
    })
if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run()