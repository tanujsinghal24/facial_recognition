import os
import cv2
import numpy as np
import pickle
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(BASE_DIR,'images')

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
x_train=[]
y_train=[]
curr_id=0
label_ids={}

for root,dirs,files in os.walk(img_dir):
	for file in files:
		if file.endswith('png') or file.endswith('jpg'):
			path=os.path.join(root,file)
			label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
			print(label)
			if label not in label_ids:
				label_ids[label]=curr_id
				curr_id+=1
			size=[500,500]
			image=Image.open(path).convert('L')
			image=image.resize(size,Image.ANTIALIAS)
			image=np.array(image,'uint8')
			curr_img_id=label_ids[label]
			faces=face_cascade.detectMultiScale(image)
			for (x,y,w,h) in faces:
				roi=image[y:y+h,x:x+w]
				x_train.append(roi)
				y_train.append(curr_img_id)
y_train=np.array(y_train)

with open("label_dict.pkl",'wb') as file:
	pickle.dump(label_ids,file)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train,y_train)
recognizer.save("recog_trained.yml")
print("training done!")