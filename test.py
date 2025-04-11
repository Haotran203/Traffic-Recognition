import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
#############################################
 
frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)
# IMPORT THE TRANNIED MODEL
model=load_model("model.h5")  ## rb = READ BYTE
 
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
def getCalssName(classNo):
    if   classNo == 0: return 'Gioi han toc do 20 km/h'
    elif classNo == 1: return 'Gioi han toc do 30 km/h'
    elif classNo == 2: return 'Gioi han toc do 50 km/h'
    elif classNo == 3: return 'Gioi han toc do 60 km/h'
    elif classNo == 4: return 'Gioi han toc do 70 km/h'
    elif classNo == 5: return 'Gioi han toc do 80 km/h'
    elif classNo == 6: return 'Ket thuc gGoi han toc do 80 km/h'
    elif classNo == 7: return 'Gioi han toc do 100 km/h'
    elif classNo == 8: return 'Gioi han toc do 120 km/h'
    elif classNo == 9: return 'Cam vuot'
    elif classNo == 10: return 'Cam vuot xe co trong tai tren 3,5 tan'
    elif classNo == 11: return 'Quyen uu tien tai giao lo tiep theo'
    elif classNo == 12: return 'Duong uu tien'
    elif classNo == 13: return 'Giao nhau voi đuong uu tien'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'Cam phuong tien giao thong'
    elif classNo == 16: return 'Cam xe tren 3.5 tân'
    elif classNo == 17: return 'Cam vao'
    elif classNo == 18: return 'Chu y chung'
    elif classNo == 19: return 'Duong cong nguy hiem ben trai'
    elif classNo == 20: return 'Duong cong nguy hiem ben phai'
    elif classNo == 21: return 'Duong cong kep'
    elif classNo == 22: return 'Duong go ghe'
    elif classNo == 23: return 'Duong tron'
    elif classNo == 24: return 'Duong hep ben phai'
    elif classNo == 25: return 'Cong truong dang thi cong'
    elif classNo == 26: return 'Tin hieu giao thong'
    elif classNo == 27: return 'Nguoi di bo'
    elif classNo == 28: return 'Tre em qua duong'
    elif classNo == 29: return 'Xe đap qua duong'
    elif classNo == 30: return 'Chu y bang/tuyet'
    elif classNo == 31: return 'Dong vat hoang da qua duong'
    elif classNo == 32: return 'Het tat ca cac gioi han toc đo va vuot qua'
    elif classNo == 33: return 'Re phai phia truoc'
    elif classNo == 34: return 'Re trai phia truoc'
    elif classNo == 35: return 'Chi di thang'
    elif classNo == 36: return 'Di thang hoac re phai'
    elif classNo == 37: return 'Di thang hoac re trai'
    elif classNo == 38: return 'Di ve ben phai'
    elif classNo == 39: return 'Di ve ben trai'
    elif classNo == 40: return 'Vong xuyen bat buoc'
    elif classNo == 41: return 'Het cam vuot'
    elif classNo == 42: return 'Het cam vuot voi xe tren 3.5 tan'
while True:
    # READ IMAGE
    success, imgOrignal = cap.read()
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    predictions = model.predict(img)
    #classIndex = model.predict_classes(img)
    classIndex = np.argmax(predictions, axis=1)
    probabilityValue =np.amax(predictions)
    cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)

    k=cv2.waitKey(1) 
    if k== ord('q'):
        break

cv2.destroyAllWindows()
cap.release()