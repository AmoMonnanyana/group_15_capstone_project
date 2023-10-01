from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import os
from werkzeug.exceptions import RequestEntityTooLarge
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import metrics
from sqlalchemy import REAL

app = Flask(__name__)
app.secret_key = "Amo101"
app.config['UPLOAD_DIRECTORY'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = ['.csv']
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///metals.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db= SQLAlchemy(app)
app.app_context().push()

#Tbale for input values prior to prediction
class metal_inputs(db.Model):
        id= db.Column("id", db.Integer, primary_key=True)
        lat = db.Column(REAL)
        long = db.Column(REAL)
        cd= db.Column(REAL)
        cr = db.Column(REAL)
        ni = db.Column(REAL)
        pb = db.Column(REAL)
        zn = db.Column(REAL)
        cu = db.Column(REAL)
        co = db.Column(REAL)
       
        def __init__(self, lat, long, cd, cr, ni, pb, zn, cu, co):
             self.lat = lat
             self.long = long
             self.cd = cd
             self.cr = cr
             self.ni = ni
             self.pb = pb
             self.zn = zn
             self.cu = cu
             self.co = co


#Table for input values with the associated results             
class input_results(db.Model):
        id= db.Column("id", db.Integer, primary_key=True)
        lat = db.Column(REAL)
        long = db.Column(REAL)
        cd= db.Column(REAL)
        cr = db.Column(REAL)
        ni = db.Column(REAL)
        pb = db.Column(REAL)
        zn = db.Column(REAL)
        cu = db.Column(REAL)
        co = db.Column(REAL)
        predicted_mCdeg = db.Column(db.String())
        predicted_class = db.Column(db.String())
        
        def __init__(self, lat, long, cd, cr, ni, pb, zn, cu, co, predicted_mCdeg, predicted_class):
             self.lat = lat
             self.long = long
             self.cd = cd
             self.cr = cr
             self.ni = ni
             self.pb = pb
             self.zn = zn
             self.cu = cu
             self.co = co
             self.predicted_mCdeg = predicted_mCdeg
             self.predicted_class = predicted_class
             

#Table for file upload data and the associated results
class file_data(db.Model):
        id= db.Column("id", db.Integer, primary_key=True)
        lat = db.Column(REAL)
        long = db.Column(REAL)
        cd= db.Column(REAL)
        cr = db.Column(REAL)
        ni = db.Column(REAL)
        pb = db.Column(REAL)
        zn = db.Column(REAL)
        cu = db.Column(REAL)
        co = db.Column(REAL)
        predicted_mCdeg = db.Column(db.String())
        predicted_class = db.Column(db.String())

        def __init__(self, lat, long, cd, cr, ni, pb, zn, cu, co, predicted_mCdeg, predicted_class):
             self.lat = lat
             self.long = long
             self.cd = cd
             self.cr = cr
             self.ni = ni
             self.pb = pb
             self.zn = zn
             self.cu = cu
             self.co = co
             self.predicted_mCdeg = predicted_mCdeg
             self.predicted_class = predicted_class


#Decoded classes based on ann_c model
classes = ['very low contamination', 
           'low contamination', 
           'moderate contamination', 
           'high contamination', 
           'very high contamination', 
           'extremely high contamination', 
           'ultra-high contamination']

class_encoder = LabelEncoder()
encoded_classes = class_encoder.fit_transform(classes)
print(encoded_classes)

#Loading ann models
try:
   ann_c = tf.keras.models.load_model("ml_models/ann-c_model.h5")
   ann_r=tf.keras.models.load_model("ml_models/ann-r_model.h5")
   print("Model successfully loaded.")

except Exception as e:
    print(f"Error loading the model: {str(e)}")

@app.route("/")
def home():
    return "home page"

method = ''
#INPUT DATA METHOD

@app.route("/input", methods=['POST', 'GET'])
def input():
    
    if request.method == 'POST':
        
        lat = request.form['lat']
        long = request.form['long']
        cd = request.form['cd']
        cr = request.form['cr']
        ni = request.form['ni']
        pb = request.form['pb']
        zn = request.form['zn']
        cu = request.form['cu']
        co = request.form['co']
        
        hm = metal_inputs(lat, long, cd, cr, ni, pb, zn, cu, co)
        db.session.add(hm)
        db.session.commit()
        db.session.close()

        input_status = request.form['input_status']
        session['input_status'] =input_status

        return redirect (url_for("process_data"))
             
    else:
        session['input_status'] = ""
        return render_template("concetration_inputs.html")

          
@app.route("/process_data")
def process_data():
        method='input'
        if session['input_status'] == 'add_more':
            return redirect(url_for("input"))

        elif session['input_status'] == "done":
            old_data = input_results.query.all()
            for data_instance in old_data:
                db.session.delete(data_instance)

            inputs = metal_inputs.query.all()
            data = [(value.lat, value.long, value.cd, value.cr, value.ni, value.pb, value.zn, value.cu, value.co) for value in inputs]
            #print(data)
            input_set = pd.DataFrame(data, columns=["lat", "long", "cd", "cr", "ni", "pb", "zn", "cu", "co"])
            X = input_set.iloc[:, 2:].values
            #print(X)

            class_prediction = ann_c.predict(X)
            y_predicted_classes = np.argmax(class_prediction, axis=1)
            #print(class_prediction)
            decoded_predicted_classes = class_encoder.inverse_transform(y_predicted_classes)
            print(decoded_predicted_classes)
            reg_prediction = ann_r.predict(X)
            print(reg_prediction)

            input_set['predicted_mCdeg'] = reg_prediction
            input_set['predicted_class'] = decoded_predicted_classes
            print(input_set)
            
            data_to_insert = input_set.to_dict(orient='records')
            new_data = [input_results(**data) for data in data_to_insert]
            db.session.add_all(new_data)
            db.session.commit()
            db.session.close()
            return redirect( url_for("view", mthd=method))

#UPLOADING FILES METHOD
@app.route('/upload', methods=['POST', 'GET'])
def upload():
    try:
        if request.method == 'POST':
        
            file = request.files['file']
            extention = os.path.splitext(file.filename)[1]
            print(extention)
            if file:
                if extention not in app.config['ALLOWED_EXTENSIONS']:
                    return "File format not supported! Please upload pdf, excel or csv files."
                file.save(os.path.join(
                    app.config['UPLOAD_DIRECTORY'], 
                    secure_filename(file.filename)
               
                ))
                file_name = secure_filename(file.filename)
                return redirect(url_for('read_file', new_file=file_name))
            else:
                return render_template('upload.html')
        else:
            return render_template('upload.html')
        
    except RequestEntityTooLarge:
        return "File is too large than the 20MB limit!"

@app.route("/read_file/<new_file>")
def read_file(new_file):
    method='upload'

    old_data = file_data.query.all()
    for data_instance in old_data:
        db.session.delete(data_instance)

    data = []
    filepath = f'uploads/{new_file}'
    with open(filepath) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data.append(row)
    
    final_set = []
    for value in data[1:]:
        list= []
        for each in value:
            converted_val = float(each)
            list.append(converted_val)
            #print(type(converted_val))
        final_set.append(list)
    dataset = pd.DataFrame(final_set, columns=["lat", "long", "cd", "cr", "ni", "pb", "zn", "cu", "co"])
    X = dataset.iloc[:, 2:].values
    
    #PREDICTION
    class_prediction = ann_c.predict(X)
    y_predicted_classes = np.argmax(class_prediction, axis=1)
    #print(y_predicted_classes)
    decoded_predicted_classes = class_encoder.inverse_transform(y_predicted_classes)
    #print(decoded_predicted_classes)
    reg_prediction = ann_r.predict(X)
    print(reg_prediction)

    dataset['predicted_mCdeg'] = reg_prediction
    dataset['predicted_class'] = decoded_predicted_classes

    
    data_to_insert = dataset.to_dict(orient='records')
    new_data = [file_data(**data) for data in data_to_insert]
    db.session.add_all(new_data)
    db.session.commit()
    db.session.close()

    return redirect(url_for("view", mthd=method))

@app.route("/view/<mthd>")
def view(mthd):
    if mthd == 'input':
        results = input_results.query.all()
    elif mthd == 'upload':
         results = file_data.query.all()

    return render_template('view.html', data=results)

    
if __name__ == " __main__ ": 
    db.create_all()
    app.run(debug=True)


#Query children of a parent(example)
#parent = Parent.query.filter_by(name='John').first()
#children = parent.children