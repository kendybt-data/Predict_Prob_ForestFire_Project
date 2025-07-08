from flask import Flask, request, jsonify, render_template

import skops.io as sio
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)
app = application


#import model y scaler
model = sio.load('models/ridge_model.skops')
scaler = sio.load('models/scaler.skops')

@app.route('/')
def index():
    return render_template ('index.html')

@app.route('/predictions', methods=['GET', 'POST'])
def make_predictions():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_input_scaled = scaler.transform([[Temperature, RH, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = model.predict(new_data_input_scaled)
        
        return render_template('home.html', results=round(result[0],3))
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
    
    