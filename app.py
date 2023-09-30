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


class metals(db.Model):
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

#load the model
try:
   ann_c = tf.keras.models.load_model("ml_models/ann-c_model.h5")
   ann_r=tf.keras.models.load_model("ml_models/ann-r_model.h5")
   print("Model successfully loaded.")

except Exception as e:
    print(f"Error loading the model: {str(e)}")

@app.route("/")
def home():
    return "home page"


#Insert data to the database
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
        initial_Cdeg = 0
        initial_class = "no class"
        print(initial_Cdeg)

        hm = metals(lat, long, cd, cr, ni, pb, zn, cu, co, initial_Cdeg, initial_class)
        db.session.add(hm)
        db.session.commit()

        return redirect (url_for("view"))
    else:
        return render_template("concetration_inputs.html")


@app.route("/view")
def view():
    values = metals.query.all()
    data = [(val.lat, val.long, val.cd, val.cr, val.ni, val.pb, val.zn, val.cu, val.co) for val in values]
    dataset = pd.DataFrame(data, columns=["lat", "long", "cd", "cr", "ni", "pb", "zn", "cu", "co"])
    X = dataset.iloc[:, 2:].values

    #PREDICTION
    class_y = ann_c.predict(X)
    y_predicted_classes = np.argmax(class_y, axis=1)
    print(class_y)
    decoded_predicted_classes = class_encoder.inverse_transform(y_predicted_classes)
    print(decoded_predicted_classes)
    print(X)
    reg_prediction = ann_r.predict(X)
    #print(reg_prediction)

    #UPDATE THE PREDICTED VALUES IN DATABASE
    # add regression values 
    new_values = {}
    n= 1
    for i, value in enumerate(reg_prediction):
        each_value = str(value[0])
        new_values[n] = each_value
        n=n+1

    for record_id, new_value in new_values.items():
        record_to_update = metals.query.get(record_id)
        if record_to_update:
            record_to_update.predicted_mCdeg = new_value
    db.session.commit()

    # Add class values
    class_values = {}
    c=1
    for class_ in decoded_predicted_classes:
        class_values[c] = class_
        c = c+1
    #print(class_values)

    for class_id, class_value in class_values.items():
        class_to_update = metals.query.get(class_id)
        if class_to_update:
            class_to_update.predicted_class = class_value
    db.session.commit()
    return render_template("view.html", values = values)

#UPLOADING FILES

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

    return "Prediction successful!"

@app.route("/predict/<input_data>")
def predict(input_data):
    inputs = np.array(input_data)
    
    #print(inputs)
    print(type(inputs))
    return "prediction hanging"

if __name__ == " __main__ ": 
    db.create_all()
    app.run(debug=True)