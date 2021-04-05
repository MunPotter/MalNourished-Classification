from flask import Flask, request, jsonify
from predict import predict
import base64
import numpy
import base64
import cv2
import pickle
from flask import Flask, render_template, request, redirect, flash, url_for
import urllib.request
from werkzeug.utils import secure_filename
import os
from process_img import process_img





model = open("Malnurished_svm_model.pkl","rb")
clf2 = pickle.load(model)







UPLOAD_FOLDER = 'F:\\single photo traffic server\\server_example'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 


 


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def submit_file():
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            processed_features_array=process_img(filename)
            prediction = predict(clf2,processed_features_array)
            prediction = int(prediction[0])
            
            if prediction == 0 :
                pre = "Malnourished"
            else:
                pre = "Non Malnourished"


            # test = os.listdir(UPLOAD_FOLDER)
            # for i in test:
            #     if i.endswith(".jpg"):
            #         os.remove(os.path.join(UPLOAD_FOLDER, i))



  
            

            return jsonify({"prediction": pre})


if __name__ == "__main__":
    app.run()