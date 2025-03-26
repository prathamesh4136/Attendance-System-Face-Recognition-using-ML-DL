from collections import Iterable
import numpy as np
import imutils
import pickle
import time
import cv2
import csv

def flatten(lis): 
    for item in lis: 
        if isinstance(item, Iterable) and not isinstance(item, str):
            # Checks if the item is an Iterable and that it is not a string
            for x in flatten(item):
                # The function calls itself recursively to flatten further
                yield x 
        else: 
            yield item

embeddingFile = "C:/Users/DELL/AppData/Local/Programs/Python/Python39/30 Days/Day 18/output/embeddings.pickle"
embeddingModel = "C:/Users/DELL/AppData/Local/Programs/Python/Python39/30 Days/Day 18/openface.nn4.small2.v1.t7"
recognizerFile = "C:/Users/DELL/AppData/Local/Programs/Python/Python39/30 Days/Day 18/output/recognizer.pickle"
labelEncFile = "C:/Users/DELL/AppData/Local/Programs/Python/Python39/30 Days/Day 18/output/le.pickle"  # Ensure the file path is correct
conf = 0.5

print("[INFO] loading face detector ...")
prototxt = "C:/Users/DELL/AppData/Local/Programs/Python/Python39/30 Days/Day 18/model/deploy.prototxt"
model = "C:/Users/DELL/AppData/Local/Programs/Python/Python39/30 Days/Day 18/model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(embeddingModel)
recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

box = []
print("[INFO] starting video stream...")
cam = cv2.VideoCapture(0)
time.sleep(1.0)

while True:
    _, frame = cam.read()  # Fixed the missing '=' sign
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    detector.setInput(imageBlob)
    detections = detector.forward()

    Roll_Number = "Unknown"  # Default value for Roll Number

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            
            if fW < 20 or fH < 20:  # Fixed typo: changed 'fw' to 'fW'
                continue 
            
            # Image to blob for face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            
            # Facial features embedder input image face blob
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            with open('student.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    box = np.append(box, row)
                    name = str(name)
                    if name in row:
                        person = str(row)
                        print(name)
                listString = str(box)
                
                # If name is found in the box list, fetch the Roll Number
                if name in listString:
                    singleList = list(flatten(box))
                    listlen = len(singleList)
                    print("listlen", listlen)
                    Index = singleList.index(name)
                    print("Index", Index)
                    name = singleList[Index]
                    Roll_Number = singleList[Index + 1]  # Assign the Roll Number value
                    print(Roll_Number) 

            # Displaying Results
            text = "{} : {} : {:.2f}%".format(name, Roll_Number, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Check if the 'Esc' key was pressed to exit
        break

cam.release()
cv2.destroyAllWindows()
