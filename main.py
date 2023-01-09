import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import mediapipe as mp
from keras.models import model_from_json
import matplotlib.pyplot as plt

# from PIL import ImageGrab

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def drawgraph(name):
    # x axis values
    if (name == "DORUKHAN"):
        y = list_dorukhan
        x = frame_list_dorukhan
    elif (name == "SUDE"):
        y = list_sude
        x = frame_list_sude
    elif (name == "BERIL"):
        y = list_beril
        x = frame_list_beril
    elif (name == "CEM"):
        y = list_cem
        x = frame_list_cem

    # corresponding y axis values
   
    #x = frame_list

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('Frame Time Graph')
    # naming the y axis
    plt.ylabel('Point Graph')

    # giving a title to my graph
    plt.title(name + "'s Performance")

    # function to show the plot
    plt.show()

def veriyazma(name, list):
    dosya = open('veriler.txt', 'w')
    list_deneme = list
    dosya.write(name + " : ")
    for isim in list_deneme:
        dosya.write(str(isim) + ",")
    dosya.write("\n")

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print('Encoding Complete')

frame_counter = 0

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.5)

emotion_dict = {0: "happy", 1: "neutral", 2: "sad", 3: "surprise"}
emotion_index = 0

json_file = open("model_architecture\\emotion_model.json")
json_file_readed = json_file.read()
model = model_from_json(json_file_readed)
model.load_weights("model_weights\\saved_model")
print('Together Complete')

beril_point = 0
emin_point = 0

points = {'BERIL':1000,'DORUKHAN':1000,'EMIN':1000,'SUDE':1000, 'YIGIT':1000, 'CEM':1000}

list_dorukhan = []
list_beril = []
list_sude =[]
list_cem =[]
frame_list_dorukhan = []
frame_list_beril = []
frame_list_sude = []
frame_list_cem = []


cap = cv2.VideoCapture("sesisleme.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
#cap = cv2.VideoCapture(0)

while True:
    yawn=False
    success, img = cap.read()
# img = captureScreen()
   
    #img = cv2.flip(img, 1)
    try:
        grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        break
    #cv2.imshow('Face Recognition', grayimage)
    
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
  
   

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            current_point = points[name]

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1-50, y1-50), (x2+50, y2+50), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 + 70), (x2, y2+40), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 +56), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            
            try: 
               #1print("here!")
               crop_img = grayimage[y1-50:y2+50, x1-50:x2+50]
               #cv2.imshow('Face Recognition', crop_img)
               crop_img2 = img[y1-50:y2+50, x1-50:x2+50]
               #1print("cropped")
               #cv2.imshow('Face Recognition', crop_img2)
               
               
               results = face_mesh.process(crop_img2)
               #1print("processed")
               img_h, img_w, img_c = crop_img2.shape
               
               #1print("here3")
               if results.multi_face_landmarks:
                   for face_landmarks in results.multi_face_landmarks:
                       
                       
                       mp_drawing.draw_landmarks(
                           image=crop_img2,
                           landmark_list=face_landmarks,
                           connections=mp_face_mesh.FACEMESH_CONTOURS,
                           landmark_drawing_spec=drawing_spec,
                           connection_drawing_spec=drawing_spec)
                       
                       
                       face_2d = []
                       yawn = []
                       position = []
                       upDown = []
                       eyes = []
                       for idx, lm in enumerate(face_landmarks.landmark):
                           
                            #sağ sol
                            if idx == 50 or idx == 93 or idx == 280 or idx == 323 or idx == 10 or idx == 298 or idx == 284:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                            
                                # Get the 2D Coordinates
                                position.append([x, y])
                            
                            if idx == 288 or idx == 435:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                upDown.append([x,y])
                            
                            if idx == 0 or idx == 13 or idx == 14 or idx == 15 or idx == 1 or idx == 18:
                                x, y = int(lm.x * img_w), int(lm.y * img_h)

                                # Get the 2D Coordinates
                                yawn.append([x, y])
                                
                            if idx == 27 or idx ==159 or idx ==145:
                               x, y = int(lm.x * img_w), int(lm.y * img_h)
                               eyes.append([x,y])
                                
                                
                         
                       if upDown[1][1]>upDown[0][1]:
                           text = "Looking Down"
                           current_point -= 10
                       elif position[1][0]<position[2][0]:
                           text = "Looking Left"
                           current_point -= 10
                       elif position[6][0]<position[3][0]:
                           text = "Looking Right"
                           current_point -= 10
                       elif position[5][1]<position[4][1]:
                           text = "Looking Up"
                           current_point -= 10
                       else:
                           text = "Forward"
                           current_point += 1
                       

                       cv2.putText(img, text, (x1 + 200, y2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 255, 0), 2)

                       
                       if (((yawn[2][1]-yawn[3][1])<(yawn[1][1]-yawn[2][1]))&((eyes[2][1]-eyes[1][1])*1.1>(eyes[0][1]-eyes[2][1]))):
                           text2 = "Yawning"
                           current_point -= 10
                           yawn=True
                       
                       else:
                           text2 = ""
                           yawn=False

                       cv2.putText(img, text2, (x1 + 200, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                                   2) #(yawn[0][0] + 10, yawn[0][1] + 10)

                       
                       if (frame_counter % 8 == 0):
                           #1print("frame counter")
                           resized_img = np.expand_dims(np.expand_dims(cv2.resize(crop_img, (48, 48)), -1), 0)
                           predictions = model.predict(resized_img)
                           emotion_index = int(np.argmax(predictions))
                           if (yawn):
                               print("yes yawn")
                               emotion_index = 1
                       cv2.putText(img, emotion_dict[emotion_index], (x1 + 200, y2 - 80),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
                       if (emotion_dict[emotion_index] == "happy"):
                            current_point += 3
                       elif (emotion_dict[emotion_index] == "sad"):
                            current_point -= 3
                       elif (emotion_dict[emotion_index] == "surprise"):
                            current_point += 1
                       
                       points[name] = current_point
                       cv2.putText(img, f"{points[name]:.2f}", (x1 + 200, y2 - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                                   2)  # (yawn[0][0] + 10, yawn[0][1] + 10)
                       if(name == "DORUKHAN"):
                           list_dorukhan.append(current_point)
                           frame_list_dorukhan.append(frame_counter)
                       elif (name == "SUDE"):
                           list_sude.append(current_point)
                           frame_list_sude.append(frame_counter)
                       elif (name == "BERIL"):
                           list_beril.append(current_point)
                           frame_list_beril.append(frame_counter)
                       elif (name == "CEM"):
                           list_cem.append(current_point)
                           frame_list_cem.append(frame_counter)

                       print(list_beril)
                       
                       #1print("resized")
                       #resized_img /= 255
                       #1print("255")
                      
                       
                       
                           
                          
               
            except Exception:
               pass 
            
            markAttendance(name)
    img = cv2.resize(img, (1080, 720))
    #cv2.imshow('Face Recognition', crop_img2)
    
    cv2.imshow('Face Recognition', img)
    frame_counter += 1
    #frame_list.append(frame_counter)
    if cv2.waitKey(5) & 0xFF == 27:
        break
veriyazma("BERIL", list_beril)
veriyazma("FRBERIL", frame_list_beril)

veriyazma("DORUKHAN", list_dorukhan)
veriyazma("FRDORUKHAN", frame_list_dorukhan)

veriyazma("CEM", list_cem)
veriyazma("FRCEM", frame_list_cem)

veriyazma("SUDE", list_sude)
veriyazma("FRSUDE", frame_list_sude)



drawgraph("BERIL")
drawgraph("DORUKHAN")
drawgraph("CEM")
drawgraph("SUDE")

cap.release()
cv2.destroyAllWindows()