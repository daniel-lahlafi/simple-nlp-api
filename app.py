from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from transformer import question_transformer, sentiment_transformer
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/question-extraction', methods=['POST'])
def question_extraction():
    post_data = request.get_json()

    question = post_data['question']
    context = post_data['context']

    return jsonify({
        'question': question,
        'context': context,
        'result': question_transformer(question, context)
    })

@app.route('/api/sentiment-extraction', methods=['POST'])
def sentiment_extraction():
    post_data = request.get_json()

    text = post_data['text']

    return jsonify({
        'text': text,
        'result': sentiment_transformer(text)
    })

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
