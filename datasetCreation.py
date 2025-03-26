import imutils
import time
import cv2
import csv
import os

cascade = 'C:/Users/DELL/AppData/Local/Programs/Python/Python39/30 Days/Day 18/haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

Name = str(input("Enter your Name: "))  
Roll_Number = int(input("Enter your Roll_Number : ")) 
dataset = 'dataset' 
sub_data = Name 
path = os.path.join(dataset, sub_data) 

os.makedirs(path, exist_ok=True)
    
info = [str(Name), str(Roll_Number)] 
with open('student.csv', 'a') as csvFile: 
     write = csv.writer(csvFile) 
     write.writerow(info) 
csvFile.close()

print("Starting video stream...") 
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open video device.")
    exit()

time.sleep(2.0)
total = 0

while total < 50:
    print(total)
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    img = imutils.resize(frame, width=400)
    rects = detector.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rects: 
         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
         p = os.path.sep.join([path, "{}.png".format(str(total).zfill(5))]) 
         cv2.imwrite(p, img) 
         total += 1
         
    cv2.imshow("Frame", frame) 
    key = cv2.waitKey(1) & 0xFF 
    if key == ord("q"): 
       break
    
cam.release() 
cv2.destroyAllWindows()
