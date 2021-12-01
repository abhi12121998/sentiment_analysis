import numpy as np
import flask
import pickle
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def sentiment_scores(sentence):
    model = SentimentIntensityAnalyzer()
    sentiment_dict = model.polarity_scores(sentence)
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        return "positive"
 
    elif sentiment_dict['compound'] <= - 0.05 :
        return "negative"
 
    else :
        return "neutral"

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form.to_dict()
        sentence = sentence['tweet']
        prediction = sentiment_scores(sentence)
        return render_template("result.html", prediction= prediction )




if __name__ == "__main__":
    app.run(debug=True, port=8000)