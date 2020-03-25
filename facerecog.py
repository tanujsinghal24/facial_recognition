import cv2
import os
import numpy as np
import pickle


face_cascade= cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_eye.xml")
 
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recog_trained.yml")
labels={}
with open("label_dict.pkl","rb") as file:
	labels=pickle.load(file)
labels={value:key for key,value in labels.items()}
cap=cv2.VideoCapture(0)
cap.set(3,400)
cap.set(4,400)
while True:
	ret,frame = cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	face=face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
	for (x,y,w,h) in face:
		roi=gray[y:y+h,x:x+w]
		roi_c=frame[y:y+h,x:x+h]
		curr_img_id ,conf = recognizer.predict(roi)
		font=cv2.FONT_HERSHEY_SIMPLEX
		name=labels[curr_img_id]
		color=(210,255,240)
		stroke=3
		if conf>10 and conf<90:
			cv2.putText(frame,name,(x,y-10),font,1,color,stroke,cv2.LINE_AA)
		img_path='my_img.png'
		cv2.imwrite(img_path,roi)
		color = (255,0,0)
		stroke = 2
		cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
	eyes=eye_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
	for (x,y,w,h) in eyes:
		roi=gray[y:y+h,x:x+w]
		color = (0,255,0)
		stroke = 2
		cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()