import cv2
import cv2 as cv
import numpy as np
from keras.models import model_from_json

emotion_dict = {0 :"happy", 1: "neutral", 2: "sad", 3: "surprise"}

json_file = open("model_architecture\\emotion_model.json")
json_file_readed = json_file.read()
model = model_from_json(json_file_readed)
model.load_weights("model_weights\\saved_model")
cam = cv.VideoCapture(0)
frame_adjuster = 0
cv.namedWindow('output', cv2.WINDOW_NORMAL)
while(True):
    is_captured, frame = cam.read()
    if is_captured == False:
        break

    detector = cv.CascadeClassifier('face_detection\\haarcascade_frontalface_default.xml')
    gray_image = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray_image, scaleFactor = 1.05, minNeighbors=5, minSize=(100,100))
    for (x,y,z,t) in faces:
        cv.rectangle(frame, (x-20, y-20), (x+z+20, y+t+20), color=(255, 0, 0), thickness=5)
        face = gray_image[y-20: y+t+20, x-20:x+z+20]
        try:
            resized_img = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)
            resized_img /= 255
        except Exception as e:
            if type(resized_img) == 'NoneType':
                continue
        predictions = model.predict(resized_img)
        emotion_index = int(np.argmax(predictions))
        cv2.putText(frame,"{}".format(emotion_dict[emotion_index]), org=(x-20,y-40),fontFace= cv.FONT_HERSHEY_SIMPLEX,thickness= 3,fontScale=1,color=(255,0,0))
    frame_adjuster += 1
    cv.imshow('output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv.destroyAllWindows()
