import time
import threading
import math
import cv2
import dlib

#classifier (choose one cars.xml or myhaar.xml)
cascadeClassifier = cv2.CascadeClassifier('myhaar.xml')

#video source
cap = cv2.VideoCapture('uhuy4.mp4')
# video = cv2.VideoCapture(0)

#count ppm to estimate car movement
def estimateSpeed(loc1, loc2):
	#you can read full formulas explanation here (https://arxiv.org/pdf/1912.00455.pdf)
	pixels = math.sqrt(math.pow(loc2[0] - loc1[0], 2) + math.pow(loc2[1] - loc1[1], 2))
	#this ppm will be different for any different road type
	ppm = 7.6
	meters = pixels / ppm
	print("pixels=" + str(pixels), "meters=" + str(meters))
	fps = 20
	speed = meters * fps * 3.6
	return speed
#detect multiobject and give carID for each detected object
def multipleObjectTrack():
	#setup bbox color and fps counter
	rectColor = (0, 0, 255)
	frameCounter = 0
	currentCarID = 0
	fps = 0
	#assign neccesary variable
	carTracker = {}
	carNumbers = {}
	carloc1 = {}
	carloc2 = {}
	carSpeed = [None] * 1000
	
#start looping
	while True:
		startTime = time.time()
		ret, image = cap.read()
		#resize your captured image
		image = cv2.resize(image, (720, 720))
		resultImage = image.copy()
		
		frameCounter = frameCounter + 1
		#setup variable to delete used car ID
		carIDtoDelete = []

		#give ID to each car
		for carID in carTracker.keys():
			trackingQuality = carTracker[carID].update(image)
			
			#if detected car more than 7, delete from first detected car(first carID)
			if trackingQuality < 7:
				carIDtoDelete.append(carID)
				
		for carID in carIDtoDelete:
			carTracker.pop(carID, None)
			carloc1.pop(carID, None)
			carloc2.pop(carID, None)
		
		if not (frameCounter % 10):
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			cars = cascadeClassifier.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
			#setup bbox
			for (_x, _y, _w, _h) in cars:
				x = int(_x)
				y = int(_y)
				w = int(_w)
				h = int(_h)
			
				x_bar = x + 0.5 * w
				y_bar = y + 0.5 * h
				
				matchCarID = None
			
				for carID in carTracker.keys():
					trackedPos = carTracker[carID].get_position()
					
					tracked_X = int(trackedPos.left())
					tracked_Y = int(trackedPos.top())
					tracked_W = int(trackedPos.width())
					tracked_H = int(trackedPos.height())
					
					tracked_X_bar = tracked_X + 0.5 * tracked_W
					tracked_Y_bar = tracked_Y + 0.5 * tracked_H
				
					if ((tracked_X <= x_bar <= (tracked_X + tracked_W)) and (tracked_Y <= y_bar <= (tracked_Y + tracked_H)) and (x <= tracked_X_bar <= (x + w)) and (y <= tracked_Y_bar <= (y + h))):
						matchCarID = carID
				
				if matchCarID is None:					
					tracker = dlib.correlation_tracker()
					tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
					
					carTracker[currentCarID] = tracker
					carloc1[currentCarID] = [x, y, w, h]

					currentCarID = currentCarID + 1
		#create line
		cv2.line(resultImage,(0,480),(1280,480),(255,255,0),2)
		
		#start track each detected car
		for carID in carTracker.keys():
			trackedPos = carTracker[carID].get_position()
					
			tracked_X = int(trackedPos.left())
			tracked_Y = int(trackedPos.top())
			tracked_W = int(trackedPos.width())
			t_h = int(trackedPos.height())
			
			cv2.rectangle(resultImage, (tracked_X, tracked_Y), (tracked_X + tracked_W, tracked_Y + t_h), rectColor, 4)

			carloc2[carID] = [tracked_X, tracked_Y, tracked_W, t_h]
		
		endTime = time.time()
		
		if not (endTime == startTime):
			fps = 1.0/(endTime - startTime)
		#put text
		cv2.putText(resultImage, 'FPS: ' + str(int(fps)), (620, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


		for i in carloc1.keys():	
			if frameCounter % 1 == 0:
				[x1, y1, w1, h1] = carloc1[i]
				[x2, y2, w2, h2] = carloc2[i]
		
				carloc1[i] = [x2, y2, w2, h2]

				if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
					if (carSpeed[i] == None or carSpeed[i] == 0) and y1 >= 275 and y1 <= 285:
						carSpeed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])


					if carSpeed[i] != None and y1 >= 180:
						cv2.putText(resultImage, str(int(carSpeed[i])) + " km/jam", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
						cv2.putText(resultImage, str(i), (int(x1 + 40), int(y1-40)),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
						print ('carID ' + str(i) + ': SpeedEstimation ' + str("%.2f" % round(carSpeed[i], 0)) + ' km/jam.\n')

		cv2.imshow('result', resultImage)

		if cv2.waitKey(33) == 27:
			break
	
	cv2.destroyAllWindows()

if __name__ == '__main__': 
	multipleObjectTrack()
	


