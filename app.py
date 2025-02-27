from flask import Flask, request, render_template, jsonify
from sentiment_analysis import predict_sentiment  # Import the predict_sentiment function from model file

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_sentiment(text)  # Use the imported predict_sentiment function
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)