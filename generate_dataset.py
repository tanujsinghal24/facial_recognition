import cv2
import os
import pickle
# face= cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)
name = input("Enter name of person in image:")
print("starting image capture")
cnt=0
face_cascade= cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR,'images/'+name+"/")):
	os.mkdir(os.path.join(BASE_DIR,'images/'+name+"/"))
img_dir=os.path.join(BASE_DIR,'images/'+name+"/")
while(True):
	ret,frame =cap.read()
	
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
	for (x,y,w,h) in faces:
		roi=frame[y:y+h,x:x+w]
		cv2.imwrite(os.path.join(img_dir ,str(cnt)+".png"),frame)
		print("img written")
		cnt+=1
		color = (0,255,0)
		stroke = 2
		cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
	cv2.imshow("Capturing Images",frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
	if cnt > 100:
		break
cap.release()
cv2.destroyAllWindows()