from scipy.spatial import distance 
from imutils import face_utils 
import imutils 
import dlib
import threading
import playsound
import cv2 

def sound_alarm(path):
	playsound.playsound(path)


def eye_aspect_ratio(eye):
	vertical1 = distance.euclidean(eye[1], eye[5])
	vertical2 = distance.euclidean(eye[2], eye[4])
	horizontal = distance.euclidean(eye[0], eye[3])
	ear = (vertical1 + vertical2) / (2.0 * horizontal)
	return ear

def main():
	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor("/home/navya/Documents/IP/Drowsiness_Detection/shape_predictor_68_face_landmarks.dat") 
	threshold=0.25
	max_closed_eye_time=20
	
	(left1, left2) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(right1, right2) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	cap=cv2.VideoCapture(0) 
	
	flag=0
	ALARM_ON = False
	while True:
		ret, frame=cap.read() 
		frame = imutils.resize(frame, width=900) 
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
		subjects = detect(gray, 0) 
		for subject in subjects:
			faceShape=predict(gray, subject)
			faceShape=face_utils.shape_to_np(faceShape) 
			(l,t,r,b) = face_utils.rect_to_bb(subject)		
			cv2.rectangle(frame, (l,t),(r+l,b+t),(0,255,0),2) 
			for (x,y) in faceShape:
				cv2.circle(frame, (x,y), 1, (255,255,0), -1) 
			leftEye = faceShape[left1:left2]
			rightEye = faceShape[right1:right2]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			EAR = (leftEAR + rightEAR) / 2.0
			leftHull = cv2.convexHull(leftEye)
			rightHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)
			if EAR < threshold:
				flag += 1
				print (flag)
				if flag >= max_closed_eye_time:
	    				if not ALARM_ON:
	    					ALARM_ON = True
						t=threading.Thread(target=sound_alarm, args=("/home/navya/Documents/IP/alarm.wav",))
						t.start()
						cv2.putText(frame, "DROWSY", (380, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
						print ("Drowsy")
			else:
				flag = 0
				ALARM_ON=False
		cv2.imshow("Frame", frame) 
		key = cv2.waitKey(1)
		if key == ord("q"):
			break
	cv2.destroyAllWindows() 
	cap.stop()
	cap.release() 

main()