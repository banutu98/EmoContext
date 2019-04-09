from flask import Flask, render_template, request

from neural_network import evaluate_conversation
from utils import parse_file

app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/emocontext", methods=['GET'])
def emocontext():
    test_data = parse_file.get_sentences_by_label("../input_data/test.txt")
    happy = test_data[0]
    sad = test_data[1]
    angry = test_data[2]
    other = test_data[3]
    return render_template('emocontext.html', happy=happy, sad=sad, angry=angry, other=other)


@app.route('/check', methods=['POST'])
def check():
    result = dict()
    result['replica1'] = request.form['replica1']
    result['replica2'] = request.form['replica2']
    result['replica3'] = request.form['replica3']
    test_data = parse_file.get_sentences_by_label("../input_data/test.txt")
    happy = test_data[0]
    sad = test_data[1]
    angry = test_data[2]
    other = test_data[3]
    output = evaluate_conversation([sentence for sentence in result.values()], '../models/final_model_adadelta_88.h5')
    return render_template('emocontext.html', label=output, happy=happy, sad=sad, angry=angry, other=other)


@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(use_reloader=True)
