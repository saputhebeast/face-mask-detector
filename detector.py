import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2

my_model = load_model('mymodel.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        while self.video.isOpened():
            success1, img = self.video.read()
            face = face_cascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 4)
            for(x, y, w, h) in face:
                face_img = img[y:y+h, x:x+w]
                cv2.imwrite('temp.jpg', face_img)
                test_image = image.load_img('temp.jpg', target_size = (150, 150, 3))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                prediction = my_model.predict(test_image)[0][0]
                
                if prediction == 1:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                    cv2.putText(img, 'NO MASK', ((x+w)//2+30, (y+h)//4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(img, 'MASK', ((x+w)//2+30, (y+h)//4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    
            success2, return_image = cv2.imencode('.jpg', img)
            return_image = return_image.tobytes()
            return return_image