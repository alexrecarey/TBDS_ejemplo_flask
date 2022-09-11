import csv

from flask import Flask, flash, jsonify, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import pygit as git

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'pdf', 'txt'}
# In a real app, the secret key would be in a separate config file and not visible in the source control
app.config['SECRET_KEY'] = '\xf7\xc4\xd3:\xdfH\x84h\xcd\xa9\xa5t\xbe\x13\x18L\x0e&\xf8\xc8}\xf9\xe9s'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Route for the GitHub webhook

@app.route('/git_update', methods=['POST'])
def git_update():
    repo = git.Repo('./TBDS_ejemplo_flask')
    origin = repo.remotes.origin
    repo.create_head('main',
                     origin.refs.main).set_tracking_branch(origin.refs.main).checkout()
    origin.pull()
    return '', 200


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=['POST'])
def predict():
    print("testing")
    if 'predictionfile' not in request.files:
        flash('No file part')
        return redirect(url_for(home))
    file = request.files['predictionfile']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for(home))
    if file and allowed_file(file.filename):
        #filename = secure_filename(file.filename)
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        datos = pd.read_csv(file)
        model = pickle.load(open('ad_model.pkl', 'rb'))
        tv = request.args.get('tv', None)
        radio = request.args.get('radio', None)
        newspaper = request.args.get('newspaper', None)

        prediction = model.predict(datos)

        #return render_template("results.html", prediction)

        return jsonify({'predictions': prediction.tolist()})

#
# return '''
#         <!doctype html>
#         <title>Upload new File</title>
#         <h1>Upload new File</h1>
#         <form method=post enctype=multipart/form-data>
#           <input type=file name=file>
#           <input type=submit value=Upload>
#         </form>
#         '''


@app.route('/api/v1/predict', methods=['GET'])
def predict_original():

    model = pickle.load(open('ad_model.pkl','rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
    
    return jsonify({'predictions': prediction[0]})

#@app.route('/api/v1/retrain', methods=['PUT'])
#def retrain():
#    data = pd.read_csv('data/Advertising.csv', index_col=0)
#
#    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
#                                                    data['sales'],
#                                                    test_size = 0.20,
#                                                    random_state=42)
#
#    model = Lasso(alpha=6000)
#    model.fit(X_train, y_train)
#
#    pickle.dump(model, open('ad_model.pkl', 'wb'))
#
#    return "Model retrained. New evaluation metric RMSE: " + str(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
if __name__ == "__main__":
    app.run()