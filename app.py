from flask import Flask, jsonify, request

from controller.analyze_controller import InputAnalyzeController


app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    json_data = request.get_json()

    text = json_data['Text']
    dataset = json_data['Dataset']
    corpora = json_data['Corpora']

    InputAnalyzeController.train(dataset, corpora)
    sentiment = InputAnalyzeController.predict(text)
    # sentiment, confidence = model.sent_analyze(text)

    return jsonify({'sentiment': int(sentiment[0])})
    # return jsonify({'sentiment': sentiment, 'confidence': confidence})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)
