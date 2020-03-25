import cv2
import numpy as numpy

file='video.avi'
frames_per_sec=64
res='720p'

def change_res(cap,width,height):
	cap.set(3,width)
	cap.set(4,height)


STD_DIMENSIONS={
	"720p":(1280,720)
}

def get_dimensions(cap,res='720p'):
	width,height=STD_DIMENSIONS['720p']
	if res in STD_DIMENSIONS:
		width,height=STD_DIMENSIONS[res]
	change_res(cap,width,height)
	return width,height


cap=cv2.VideoCapture(0)
width,height=get_dimensions(cap)
out = cv2.VideoWriter(file,cv2.VideoWriter_fourcc(*'XVID'),frames_per_sec,(width,height))
while(True):
	ret, frame = cap.read()
	out.write(frame)
	cv2.imshow("frame",frame)
	if cv2.waitKey(20) & 0xff == ord('q'):
		break
cap.release()
out.release()
cap.destroyAllWindows()