from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

MODEL_PATH ='model.h5'

model = load_model(MODEL_PATH)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img
def getClassName(classNo):
    if   classNo == 0: return 'Giới hạn tốc độ 20 km/h'
    elif classNo == 1: return 'Giới hạn tốc độ 30 km/h'
    elif classNo == 2: return 'Giới hạn tốc độ 50 km/h'
    elif classNo == 3: return 'Giới hạn tốc độ 60 km/h'
    elif classNo == 4: return 'Giới hạn tốc độ 70 km/h'
    elif classNo == 5: return 'Giới hạn tốc độ 80 km/h'
    elif classNo == 6: return 'Kết thúc giới hạn tốc độ 80 km/h'
    elif classNo == 7: return 'Giới hạn tốc độ 100 km/h'
    elif classNo == 8: return 'Giới hạn tốc độ 120 km/h'
    elif classNo == 9: return 'Cấm vượt'
    elif classNo == 10: return 'Cấm vượt xe có trọng tải trên 3,5 tấn'
    elif classNo == 11: return 'Quyền ưu tiên tại giao lộ tiếp theo'
    elif classNo == 12: return 'Đường ưu tiên'
    elif classNo == 13: return 'Giao nhau với đường ưu tiên'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'Cấm phương tiện giao thông'
    elif classNo == 16: return 'Cấm xe trên 3.5 tấn'
    elif classNo == 17: return 'Cấm vào'
    elif classNo == 18: return 'Chú ý chung'
    elif classNo == 19: return 'Đường cong nguy hiểm bên trái'
    elif classNo == 20: return 'Đường cong nguy hiểm bên phải'
    elif classNo == 21: return 'Đường cong kép'
    elif classNo == 22: return 'Đường gồ ghề'
    elif classNo == 23: return 'Đường trơn'
    elif classNo == 24: return 'Đường hẹp bên phải'
    elif classNo == 25: return 'Công trường đang thi công'
    elif classNo == 26: return 'Tín hiệu giao thông'
    elif classNo == 27: return 'Người đi bộ'
    elif classNo == 28: return 'Trẻ em qua đường'
    elif classNo == 29: return 'Xe đạp qua đường'
    elif classNo == 30: return 'Chú ý băng/ tuyết'
    elif classNo == 31: return 'Động vật hoang dã qua đường'
    elif classNo == 32: return 'Hết tất cả các giới hạn tốc độ và vượt qua'
    elif classNo == 33: return 'Rẽ phải phía trước'
    elif classNo == 34: return 'Rẽ trái phía trước'
    elif classNo == 35: return 'Chỉ đi thẳng'
    elif classNo == 36: return 'Đi thẳng hoặc rẽ phải'
    elif classNo == 37: return 'Đi thẳng hoặc rẽ trái'
    elif classNo == 38: return 'Đi về bên phải'
    elif classNo == 39: return 'Đi về bên trái'
    elif classNo == 40: return 'Vòng xuyến bắt buộc'
    elif classNo == 41: return 'Hết cấm vượt'
    elif classNo == 42: return 'Hết cấm vượt với xe trên 3.5 tấn'


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    preds = getClassName(classIndex)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
