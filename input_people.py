import numpy as np
import cv2
import argparse
import os
from random import randint
parser = argparse.ArgumentParser()
parser.add_argument("name", help = "name of person need to add")
args = parser.parse_args()

directory = "./data/data"+args.name

im_arr = []

if not os.path.exists(directory):
    os.makedirs(directory)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        im_arr.append(img)
            
        
   
# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
cap = cv2.VideoCapture(0)

while(1):
    ret, img = cap.read()
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
for i in range(len(im_arr)):
    cv2.imwrite(directory+'/'+str(i)+'.jpg',im_arr[i])
