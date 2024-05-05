from flask import Flask, render_template, jsonify,request,Response
import pyttsx3
import speech_recognition as sr
from threading import Thread
import cv2
import dlib 
import imutils 
from scipy.spatial import distance as dist 
from imutils import face_utils 
import time
import smtplib
from cvzone.HandTrackingModule import HandDetector 
import mediapipe as mp
import numpy as np


app = Flask(__name__)

Name , Email = '',''


@app.route('/')
def Login():
    return render_template('Login.html')

@app.route('/index',methods=['POST'])
def Sample():
    global Name,Email
    Name = request.form['name']
    Email = request.form['email']
    return render_template('index.html')

@app.route('/blind_only',methods=['POST'])
def blind_only():
    return render_template('step_2_voice.html')

@app.route('/blind_nd_dumb',methods=['POST','GET'])
def blind_nd_dumb():
    return render_template('step_1_hand.html')

@app.route('/disable_nd_blind',methods=['POST'])
def disable_nd_blind():
    return render_template('step_2_voice.html')

@app.route('/disable_nd_deaf',methods=['POST'])
def disable_nd_deaf():
    return render_template('step_2_voice.html')

@app.route('/deaf_nd_dumb_nd_disable',methods=['POST'])
def deaf_nd_dumb_nd_disable():
    return render_template('step_2_eye.html')

@app.route('/deaf_nd_dumb',methods=['POST'])
def deaf_nd_dumb():
    return render_template('step_1_hand.html')


@app.route('/voice_front',methods=['POST'])
def voice_front():
    return render_template('voice_front.html')

@app.route('/fingerCounter',methods=['POST'])
def fingerCounter():
    data = detect_hand_gesture()
    return render_template('step_3.html',data = data)
    

@app.route('/eye_gesture',methods=['POST'])
def eye_gesture():
    data = detect_eye_gesture()
    return render_template('step_3.html',data = data)


@app.route('/step_3.html')
def step_3():
    global Name,Email
    drink = str(request.args.get('data'))
    drink = drink.lower()
    if 'coffee' in drink:
        drink = 'coffee'
    else:
        drink = 'tea'
    
    send_mail(Name,Email,drink)
    return render_template('step_3.html', data=drink)


@app.route('/step_2_hand',methods=['POST'])
def step_2_hand():
    return render_template('step_2_hand.html')


def calculate_EAR(eye): 

	y1 = dist.euclidean(eye[1], eye[5]) 
	y2 = dist.euclidean(eye[2], eye[4]) 
	x1 = dist.euclidean(eye[0], eye[3]) 
	EAR = (y1+y2) / x1 
	return EAR 

def detect_eye_gesture():
    
    cam = cv2.VideoCapture(0) 
    blink_thresh = 0.45
    succ_frame = 2
    count_frame = 0
    blink_count = 0
    (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 
    detector = dlib.get_frontal_face_detector() 
    landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 
    start_time = time.time()
    time_limit = 20
    end_time = start_time + time_limit
    while time.time()<end_time:

        ret, frame = cam.read() 
        frame = imutils.resize(frame, width=640) 
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(img_gray) 
        for face in faces:
            shape = landmark_predict(img_gray, face)
            shape = face_utils.shape_to_np(shape)
            lefteye = shape[L_start: L_end]
            righteye = shape[R_start:R_end]
            left_EAR = calculate_EAR(lefteye)
            right_EAR = calculate_EAR(righteye)
            avg = (left_EAR+right_EAR)/2
            if avg < blink_thresh:
                count_frame += 1
            else:
                if count_frame >= succ_frame:
                    blink_count += 1
                    cv2.putText(frame, 'Blink Detected', (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                    count_frame = 0
            cv2.imshow("Video", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cam.release() 
    cv2.destroyAllWindows() 

    if blink_count == 1:
        return "TEA"
    else:
        return "COFFEE"


def detect_hand_gesture():
    
    detector = HandDetector(maxHands=1, detectionCon=0.8) 
    video = cv2.VideoCapture(0) 
    time_limit = 15
    end_time = time.time()+time_limit
    res = ''
    while time.time()<end_time: 
	    _, img = video.read() 
	    hand = detector.findHands(img, draw=False) 
	   
	    if hand: 
		    lmlist = hand[0] 
		    if lmlist: 
			    fingerup = detector.fingersUp(lmlist) 
			    if fingerup == [0, 1, 0, 0, 0]: 
				    res = "TEA"
			    if fingerup == [0, 1, 1, 0, 0]: 
				    res = "COFFEE"
	   
	    cv2.imshow("Video", img) 
	
	    if cv2.waitKey(1) & 0xFF == ord('q'): 
		    break
		
    video.release() 
    cv2.destroyAllWindows() 
    return res


def voice_reg():
    data = request.json
    voice_text = data['voice_text']
    recognizer = sr.Recognizer()
    audio_data = voice_text.encode('utf-8')
    
    try:
        result = recognizer.recognize_google(audio_data)
        processed_result = process_text(result)
        return jsonify({'result': processed_result})
    except sr.UnknownValueError:
        return jsonify({'error': 'Speech recognition could not understand audio'})
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition service error: {e}'})



def send_mail(Name,Email,drink):
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    sender_mail = "arun1772003@gmail.com"
    s.login(sender_mail, "vnfntyifwsutzrss")
    message = "Dear {}, \n\t Your are ordered the {} \n Thank you".format(str(Name),str(drink))
    receiver_mail= Email
    s.sendmail(sender_mail,receiver_mail,message)
    s.quit()    

    
    
if __name__ == '__main__':
    app.run(debug=True)
