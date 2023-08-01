from flask import Flask ,request,app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


application= Flask(__name__)
app=application

scaler = pickle.load(open('/config/workspace/model/scaler.pkl1','rb'))
model = pickle.load(open('/config/workspace/model/svc_h.pkl1','rb'))

# Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

##Route for single data point prediction
@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    result =""

    if request.method =='POST':
        sepal_length=float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width')) 


        new_data = scaler.transform([[sepal_length,sepal_width,petal_width,petal_length]])
        predict = model.predict(new_data)

        if predict[0] ==0:
            result = 'setosa'
        elif predict[0] ==1:
            result ='versicolor'
        else:
            result = 'virginica'     

        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')       


if __name__=="__main__":
    app.run(host="0.0.0.0")
