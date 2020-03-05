# Connect to local MQTT broker
# Turn on video from webcam connected to jetson
# Detect Faces convert to bytes and sent to broker

import numpy as np
import cv2
import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST="mosquitto"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="system/jetson/webcam/face"

# Function to check connection to broker in MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connection to broker: Success!")
    else:
        print("Connection to broker: Failed!")
        
# Connect to broker
client = mqtt.Client("LocalClient-02")
client.on_connect = on_connect
client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
client.loop_start()

#load the XML classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # We don't use the color information, so might as well save space
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detection and other logic goes here
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
	# cut out face from the frame and display image.
        cv2.imshow('face', face)
	rc,png = cv2.imencode('.png', face)
	msg = png.tobytes()
	# publish the face to MQTT broker
        client.publish(LOCAL_MQTT_TOPIC, payload=msg, qos=0, retain=False)

    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# Disconnect from MQTT broker
client.loop_stop()
client.disconnect()	
	
# When everything done, release the capture
cap.release()

# Close all the frames
cv2.destroyAllWindows()
