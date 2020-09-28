import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
fuel_lbl = pickle.load(open('fuel_lbl_enc', 'rb'))
trans_lbl = pickle.load(open('trans_lbl_enc', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
	features = [str(x) for x in request.form.values()] #getting datafrom html form
	year = int(features[0])
	km_driven = int(features[1])
	final = []
	final.append(year)
	final.append(km_driven)
	final.append(fuel_lbl.transform([features[2]])[0])
	final.append(trans_lbl.transform([features[3]])[0])
	prediction = model.predict([final])
	output = round(prediction[0], 2)
	return render_template('index.html', prediction_text = output)

if __name__ == '__main__':
	app.run(debug = True)
