from flask import Flask, render_template, request
import numpy as np
import pickle

import joblib
model=joblib.load(open('model.pkl','rb'))


app=Flask(__name__)
@app.route('/',methods=['GET'])
def index():
    return render_template('home.html')
@app.route('/products',methods=['POST','GET'])
def products():
    
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model.predict(final)
    if(prediction==0):
        return render_template('index.html',po="LOW")
    elif(prediction==1):
        return render_template('index.html',po="MODERATE")
    else:
        return render_template('index.html',po="HIGH")
if __name__=="__main__":
    app.run(debug=True,port=8000)