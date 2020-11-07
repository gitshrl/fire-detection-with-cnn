import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import time

model = load_model('models/v2.0/Fire-32x32-CNN-v2.h5')

cap = cv2.VideoCapture('input/sample.mp4') # replace 'input/sample.mp4' with 0 to use your webcam instead
time.sleep(2)

if cap.isOpened():
    rval, frame = cap.read()
else:
    rval = False

IMG_SIZE = 32

while True:
    rval, image = cap.read()
    
    if rval == True:
        orig = image.copy()

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))  
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        
        fire_prob = model.predict(image)[0][0]
        label = "Confidence rate: {:.2f}".format(fire_prob)
        
        if fire_prob > 0.50:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        
        cv2.putText(orig, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Output", orig)
        
        key = cv2.waitKey(10)
        if key == 27:
            break
    elif rval==False:
            break

cap.release()
cv2.destroyAllWindows()

