
#importing packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import playsound
import time
import dlib
import cv2
import my_library


class head_pose_detection:

	#constants and thresholds
	HEAD_COUNTER = 0
	FACE_COUNTER = 0
	ALARM = False
	HEAD_SUBSEQUENT_FRAMES = 50
	NO_FACE_DETECTION_SUBSEQUENT_FRAMES = 60
	PITCH_UP = 5.0
	PITCH_DOWN = -25.0
	YAW_UP = 25.0
	YAW_DOWN = -22.0
	line_pairs = [[0, 1], 
				  [1, 2], 
				  [2, 3], 
				  [3, 0],
        		  [4, 5], 
				  [5, 6], 
				  [6, 7], 
				  [7, 4],
            	  [0, 4],
				  [1, 5], 
				  [2, 6], 
				  [3, 7]]


	#head alert thread
	def sound_alarm_head(self):
		playsound.playsound("look_straight.mp3")
		time.sleep(1)
		self.HEAD_COUNTER = 0
		self.ALARM = False


	#face not detected alert thread
	def sound_alarm_face_not_detect(self):
		playsound.playsound("face_not_detected.mp3")
		time.sleep(1)
		self.FACE_COUNTER = 0
		self.ALARM = False


	#main method
	def __init__(self):

		#initializing dlib's face detector and creating the facial landmark predictor
		print("loading trained models...")
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

				#calculating head angle by calling my_library function
				reprojected_points, euler_angle = my_library.get_head_pose(coords)
				
				#if head is not straight
				if euler_angle[0, 0] > self.PITCH_UP or euler_angle[0, 0] < self.PITCH_DOWN or euler_angle[1, 0] > self.YAW_UP or euler_angle[1, 0] < self.YAW_DOWN:
					self.HEAD_COUNTER += 1
					
					#if head threshold crossed
					if self.HEAD_COUNTER >= self.HEAD_SUBSEQUENT_FRAMES:

						#triggering alarm
						if self.ALARM == False:
							self.ALARM = True
							t1 = Thread(target = self.sound_alarm_head)			
							t1.deamon = True
							t1.start()
				
				#if head is straight
				else:
					self.HEAD_COUNTER = 0
			
			#adding black pixels to the main frame
			black = np.zeros((57, 570, 3), np.uint8)
			frame = np.vstack((black, frame, black))

			#overlay the red alert screen
			overlay = frame.copy()
			cv2.rectangle(overlay, (0,57) ,(570, 485), (0, 0, 255), -1)
			
			#displaying results on the frame
			#if face is detected
			if len(rects) > 0:

				#drawing the projected cube
				for start, end in self.line_pairs:
					cv2.line(frame, (int(reprojected_points[start, 0]), int(reprojected_points[start, 1] + 43)), (int(reprojected_points[end, 0]), int(reprojected_points[end, 1] + 43)), (255, 128, 0), 2, 8)

				#if head is not straight print PITCH in red color
				if euler_angle[0, 0] > self.PITCH_UP or euler_angle[0, 0] < self.PITCH_DOWN:
					cv2.putText(frame, "PITCH: " + "{:.2f}".format(euler_angle[0, 0]), (18, 523), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 0, 255), 2)
				
				#if head is straight print PITCH in green color
				else:
					cv2.putText(frame, "PITCH: " + "{:.2f}".format(euler_angle[0, 0]), (18, 523), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 255, 0), 2)

				#if head is not straight print YAW in red color
				if euler_angle[1, 0] > self.YAW_UP or euler_angle[1, 0] < self.YAW_DOWN:
					cv2.putText(frame, "YAW: " + "{:.2f}".format(euler_angle[1, 0]), (310, 523), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 0, 255), 2)
				
				#if head is straight print YAW in green color
				else:
					cv2.putText(frame, "YAW: " + "{:.2f}".format(euler_angle[1, 0]), (310, 523), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (0, 255, 0), 2)

				#drawing boxes
				cv2.rectangle(frame, (15, 480) ,(278, 537), (255, 255, 255), 2)
				cv2.rectangle(frame, (292,480) ,(555, 537), (255, 255, 255), 2)

				#if head is not straight
				if euler_angle[0, 0] > self.PITCH_UP or euler_angle[0, 0] < self.PITCH_DOWN or euler_angle[1, 0] > self.YAW_UP or euler_angle[1, 0] < self.YAW_DOWN:
					
					#if head threshold crossed add red screen and print warning message
					if self.HEAD_COUNTER >= self.HEAD_SUBSEQUENT_FRAMES:
						cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
						cv2.putText(frame, "WARNING: LOOK STRAIGHT!", (72, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 2)

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
			cv2.imshow("Head Pose Detection", frame)
		
			#press 'q' to break the loop
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

		#close the window and quit
		cv2.destroyAllWindows()
		vs.stop()

#creating the class object
#head_pose = head_pose_detection()