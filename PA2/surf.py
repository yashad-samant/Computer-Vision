import cv2
import numpy as np
import math
import sys
import os

os.chdir('C:/Users/Yashad/Desktop/Books/image computation')
previousWindows = []

cap = cv2.VideoCapture('example.mov')

hessian_threshold = 2000
detector = cv2.SURF(hessian_threshold)
detector.nOctaves = 5
detector.nOctaveLayers = 1
detector.upright = True


frame = 0
while(True):
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	hkeypoints,hdescriptors = detector.detectAndCompute(gray, None)
	
	best = hkeypoints[0]
	
	x = 0
	y = 0
	x2 = 0
	y2 = 0
	while(True):
		for keypoint in hkeypoints:
			if best.response < keypoint.response:
				best = keypoint
			
		x = int(best.pt[0] - best.size / 2)
		y = int(best.pt[1] - best.size / 2)
		x2 = x + int(best.size)
		y2 = y + int(best.size)
		
		tryAgain = False
		for rect in previousWindows:
			ox = rect[0]
			oy = rect[1]
			ox2 = rect[2]
			oy2 = rect[3]
			
			x_overlap = max(0, min(x2, ox2) - max(x, ox));
			y_overlap = max(0, min(y2, oy2) - max(y, oy));
			overlapArea = x_overlap * y_overlap;
			if (overlapArea) > ((x2 - x) * (y2 - y) / 2):
				tryAgain = True
				hkeypoints.remove(best)
				best = hkeypoints[0]
				break
				
		if tryAgain == False:
			break
	
	
	kimg = cv2.drawKeypoints(img, hkeypoints, None, (250, 150, 50), 4)
	kimg = cv2.drawKeypoints(kimg, [best], None, (0, 255, 255), 4)
	cv2.rectangle(kimg, (x, y), (x2, y2), (0, 255, 255), 2)
	cv2.imshow('window2', kimg)
	
	attentionWindow = img[y:y2,x:x2]
	cv2.imshow('window3', attentionWindow)
	
	cv2.imwrite('SURF/' + str(frame) + '.png', attentionWindow)
	if(frame == 100): cv2.imwrite('ScreenShot_SURF'+ str(frame), kimg)
	
	previousWindows.append([x, y, x2, y2])
	if(len(previousWindows) > 30):
		previousWindows.pop(0)
	
	frame+=1

	print(len(hkeypoints))

	
	if cv2.waitKey(15)==13:
		break
	

cap.release()
cv2.destroyAllWindows()