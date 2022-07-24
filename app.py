from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
#import joblib
import pickle
  

import flask

app = Flask(__name__)

with open('log.bin','rb') as f:
  model = pickle.load(f) 
    
###################################################

###################################################


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    user_input = (to_predict_list['user_input']).split(" ")
    user_input = [[float(inp) for inp in user_input]]
    pred = model.predict(user_input)
    #pr =  1
    if pred[0] == 0:
        prediction = "Setosa"
        #pr = prob[0][0]
    elif pred[0] == 1:
        prediction = "Versicolor"
        #pr = prob[0][0]
    else:
        prediction = "Virginica"
    # sanity check to filter out non questions.     
    return flask.render_template('predict.html', prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)
