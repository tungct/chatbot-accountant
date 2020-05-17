from flask import *
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
    msg = 'test'
    with open('../data/qa.txt', 'r') as f:
        res = f.readlines()
    return json.dumps({
        'message': res
    })
if __name__ == '__main__':
    app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.run()