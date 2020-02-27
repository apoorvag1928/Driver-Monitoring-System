
#importing packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
import time
import dlib
import cv2
import queue
import my_library


class talking_detection:

	#constants and thresholds
	FACE_COUNTER = 0
	ALARM = False
	MAR_THRESHOLD = 0.08
	NO_FACE_DETECTION_SUBSEQUENT_FRAMES = 60
	MOUTH_QUEUE = 90


	#face not detected alert thread
	def sound_alarm_face_not_detect(self):
		playsound.playsound("face_not_detected.mp3")
		time.sleep(1)
		self.FACE_COUNTER = 0
		self.ALARM = False


	#mouth alert thread
	def sound_alarm_mar(self):
		playsound.playsound("dont_talk.mp3")
		time.sleep(2)
		self.ALARM = False


	#main method
	def __init__(self):

		#initializing dlib's face detector and creating the facial landmark predictor
		print("loading trained models...")
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

		#initializing queue for MAR average calculation 
		mar_queue = queue.Queue(maxsize= 100)
		for i in range(self.MOUTH_QUEUE):
			mar_queue.put(0.0)
		avg_mar = 0.0

		#starting the video stream
		print("starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(1.0)

		#loop until we press 'q'
		while True:

			#reading current frame, resize and filter it
			frame = vs.read()
			frame = my_library.resize(frame, width=570)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			#feeding the grayscale frame to face detector model
			rects = detector(gray)

			#if any face detected in the image
			if len(rects) > 0:

				#face detected, so make the counter value zero
				self.FACE_COUNTER = 0

				#feeding the grayscale frame and detected face to shape prediction model
				shape = predictor(gray, rects[0])

				#extracting all 68 coordinates
				coords = np.zeros((68, 2), int)
				for i in range(0, 68):
					coords[i] = (shape.part(i).x, shape.part(i).y)

				#extracting left eye, right eye and mouth coordinates
				leftEye = coords[42:48]
				rightEye = coords[36:42]
				mouth = coords[60:68]

				#calculating MAR
				mar = my_library.mouth_aspect_ratio(mouth)

				#calculating average MAR
				mar_ex = mar_queue.get()		
				mar_queue.put(mar)
				avg_mar = avg_mar + ((mar - mar_ex) / float(self.MOUTH_QUEUE))

				#if mouth threshold crossed
				if avg_mar > self.MAR_THRESHOLD:
					
					#if mouth threshold is crossed extremely
					if avg_mar > (self.MAR_THRESHOLD + 0.03):
						for i in range(self.MOUTH_QUEUE):
							mar_queue.get()
							mar_queue.put(0.0)
						avg_mar = 0.0

					#triggering alarm
					if self.ALARM == False:
						self.ALARM = True
						t3 = Thread(target = self.sound_alarm_mar)			
						t3.deamon = True
						t3.start()
			
			#adding black pixels to the main frame
			black = np.zeros((57, 570, 3), np.uint8)
			frame = np.vstack((black, frame, black))

			#overlay the red alert screen
			overlay = frame.copy()
			cv2.rectangle(overlay, (0,57) ,(570, 485), (0, 0, 255), -1)
			
			#displaying results on the frame
			#if face is detected
			if len(rects) > 0:

				#draw the mask by calling my_library function
				my_library.mask(coords, frame, 0, 57)

				#drawing eyes and mouth boundaries
				leftEyeHull = cv2.convexHull(leftEye)
				rightEyeHull = cv2.convexHull(rightEye)
				mouthHull = cv2.convexHull(mouth)
				cv2.drawContours(frame, [leftEyeHull], -1, (128, 128, 128), 2, offset = (0, 57))
				cv2.drawContours(frame, [rightEyeHull], -1, (128, 128, 128), 2, offset = (0, 57))
				cv2.drawContours(frame, [mouthHull], -1, (128, 128, 128), 2, offset = (0, 57))
				
				#drawing boxes
				cv2.rectangle(frame, (180, 480) ,(392, 537), (255, 255, 255), 2)

				#if MAR threshold crossed print MAR in red color
				if mar > self.MAR_THRESHOLD:
					cv2.putText(frame, "MAR: {:.2f}".format(mar), (200, 523), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 0, 255), 2)
				
				#if MAR threshold not crossed print MAR in green color
				else:
					cv2.putText(frame, "MAR: {:.2f}".format(mar), (200, 523), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 255, 0), 2)

				#if mouth threshold crossed add red screen and print warning message
				if avg_mar > self.MAR_THRESHOLD:
					cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
					cv2.putText(frame, "WARNING: DON'T TALK!", (107, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 2)

			#if face not detected
			else:
				self.FACE_COUNTER += 1
				
				#if face not detected threshold crossed add red screen and print warning message
				if self.FACE_COUNTER > self.NO_FACE_DETECTION_SUBSEQUENT_FRAMES:	
					cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
					cv2.putText(frame, "WARNING: FACE NOT DETECTED!", (32, 520), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 2)
					
					#triggering alarm
					if self.ALARM == False:
						self.ALARM = True
						t3 = Thread(target = self.sound_alarm_face_not_detect)			
						t3.deamon = True
						t3.start()

			#displaying frame
			cv2.imshow("Talking Detection", frame)
		
			#press 'q' to break the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

		#close the window and quit
		cv2.destroyAllWindows()
		vs.stop()

#creating the class object
#talk = talking_detection()