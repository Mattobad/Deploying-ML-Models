from flask import Flask,render_template, request
from wtforms import Form, TextField, DecimalField, validators
import pickle
import joblib
import os 
import numpy as np
import pandas as pd


app = Flask(__name__)

# class HelloForm(Form):
#     sayHello = TextField('',[validators.DataRequired()])

### loading the ml model
cur_dir = os.path.dirname(__file__)
lin_reg = joblib.load(open(os.path.join(cur_dir,'pkl_objects','lin_regssor_model.pkl'),'rb'))

#module for model prediction
def lin_regression(data_in):
    y = lin_reg.predict(data_in)
    #proba =np.max(lin_reg.predict_proba(data_in))
    return y


#Flask 
class ReviewForm(Form):
    year = DecimalField('Year of Experience',[validators.DataRequired()])
    # Feature1 = TextField('Feature1',[validators.DataRequired(), validators.length(min=15)])
    # Feature2 = TextField('Feature2',[validators.DataRequired(), validators.length(min=15)])
    # Feature3 = TextField('Feature3',[validators.DataRequired(), validators.length(min=15)])
    # Feature4 = TextField('Feature4',[validators.DataRequired(), validators.length(min=15)])


@app.route('/')
def index():
    # form = HelloForm(request.form)
    # return render_template('index.html', form=form)
    form = ReviewForm(request.form)
    return render_template('reviewform.html',form=form)


@app.route('/results',methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():

        ''' for model with feature more than one
        # feature_array =[]
        # feature_array.append(request.form['Feature1'])
        # feature_array.append(request.form['Feature2'])
        # feature_array.append(request.form['Feature3'])
        # feature_array.append(request.form['Feature4'])

        # feature_df = pd.DataFrame(feature_array)
        '''
        feature_year = []
        feature_year.append(request.form['year'])
        feature_df = pd.DataFrame(feature_year)
        y= lin_regression(feature_df)
        return render_template('results.html', content=feature_df, prediction=y)
        #return render_template('results.html')
    
    return render_template('reviewform.html', form=form)

# @app.route('/hello', methods=['POST'])

# def hello():
    # form = HelloForm(request.form)
    # if request.method == 'POST' and form.validate():
    #     name = request.form['sayHello']
    #     return render_template('hello.html', name=name)
    # return render_template('index.html',form=form)

if __name__ == '__main__':
    app.run(debug=True)