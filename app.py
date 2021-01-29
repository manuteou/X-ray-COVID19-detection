from flask import Flask,redirect, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import tensorflow as tf
import numpy as np 
from datetime import datetime
import requests
import locale 
from PIL import Image
locale.setlocale(locale.LC_ALL, '')
from detection import detection 
from crop import crop

app = Flask(__name__)


UPLOAD_FOLDER = r"C:/Users/froge/Exo_jedha/PROJET/Site/static/Analyse/"
MODEL_PATH = r"C:/Users/froge/Exo_jedha/PROJET/Site/models/model_VGG19.h5"
#MODEL_PATH = r"C:/Users/froge/Exo_jedha/PROJET/Site/models/model_VGG16.h5"
#MODEL_PATH = r"C:/Users/froge/Exo_jedha/PROJET/Site/models/model_DenseNet.h5"
IMAGE_PATH = r"Analyse/"

@app.route("/")
def index():
    # request Frensh API COVID
    try:
        r = requests.get( 
        "https://coronavirusapi-france.now.sh/FranceLiveGlobalData")
        info = r.json()
    except:
        return  render_template("index.html",date = 0, case = 0, hosp = 0, rea = 0, new = 0)  
    try:
        info_date = info['FranceGlobalLiveData'][0]['date']
    except KeyError:
        info_date = 'NaN'
    try:
        info_ccase = f"{info['FranceGlobalLiveData'][0]['casConfirmes']:n}"
    except KeyError:
        info_ccase = 'NaN'
    try:
        info_Hosp = f"{info['FranceGlobalLiveData'][0]['hospitalises']:n}"
    except KeyError:
        info_Hosp = 'NaN'
    try:
        info_rea = f"{info['FranceGlobalLiveData'][0]['reanimation']:n}"
    except KeyError:
        info_rea = 'NaN'   
    try:
        info_new = f"{info['FranceGlobalLiveData'][0]['nouvellesHospitalisations']:n}"
    except KeyError:
        info_new = 'NaN'   
    
    return  render_template("index.html",date = info_date, case = info_ccase, hosp = info_Hosp, rea = info_rea, new = info_new)
    

@app.route('/upload', methods=['GET','POST'])
def upload_files():
    # import of  VGG19'model
    model = tf.keras.models.load_model(MODEL_PATH)

    # image import
    if request.method == 'POST':
        # download image
        f = request.files['radio']
        filename = secure_filename(f.filename)
        file_path = Path(UPLOAD_FOLDER,filename)
        f.save(file_path)

        #we crop image
        c = crop()
        img = c.crop(filename)

    
        pred_crop_name = f"{filename}_crop.png"

        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img.save(Path(UPLOAD_FOLDER, pred_crop_name))

        file_path = Path(UPLOAD_FOLDER,pred_crop_name)

        #we do detection
        d = detection()
        # we import timestamp to gave an unique id
        time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        pred_img_name = f"prediction_{time}.png"
        # we use the detection script to predict
        grad_cam, preds = d.prediction_grad_cam(file_path,model)
        grad_cam.save(Path(UPLOAD_FOLDER,pred_img_name))
      
        pred_crop_name = f"prediction_{time}.png"
        
        # we return result on result page
        return render_template('result.html',prediction=preds, 
                images= IMAGE_PATH + pred_img_name,images_or = IMAGE_PATH + filename)
                
    return None


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
