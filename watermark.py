import numpy as np
import cv2
import os
file='video_water.mp4'
frames_per_sec=16
res='720p'

def change_res(cap,width,height):
	cap.set(3,width)
	cap.set(4,height)


STD_DIMENSIONS={
	"720p":(1280,720)
}

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
 
def get_dimensions(cap,res='720p'):
	width,height=STD_DIMENSIONS['720p']
	if res in STD_DIMENSIONS:
		width,height=STD_DIMENSIONS[res]
	change_res(cap,width,height)
	return width,height
path_to_mark="open/images/logo/cfe-coffee.png" 
mark = cv2.imread(path_to_mark,-1 )
cv2.imshow("watermark",mark)
mark_resized = image_resize(mark,height=50)


cap=cv2.VideoCapture(0)
width,height=get_dimensions(cap)
out = cv2.VideoWriter(file,cv2.VideoWriter_fourcc(*'H264'),frames_per_sec,(width,height))

  
while True:
	ret,frame = cap.read()
	cv2.imshow("video",frame)
	out.write(frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break
cap.release()
out.release()
cv2.destroyAllWindows()