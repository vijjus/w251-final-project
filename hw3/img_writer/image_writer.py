import paho.mqtt.client as mqtt
import cv2 as cv
import sys
import subprocess
import time
import numpy as np

# connect to local MQTT broker
LOCAL_MQTT_HOST="localhost"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="face_images"

BUCKET = "s3://w251-vijay-s3/"

count=0
def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    mqttclient.subscribe(LOCAL_MQTT_TOPIC)

def on_message(client, userdata, msg):
    global count
    print("Message received!: {}".format(len(msg.payload)))
    local_file = "image_" + str(count) + ".png"
    print("Local file: {}".format(local_file))
    count += 1
    # incoming data is a bytearray, convert it into numpy array
    gray = np.frombuffer(msg.payload, dtype=np.uint8)
    # captured image was encoded in .png format
    img = cv.imdecode(gray, flags=1)
    # write the image to a local file
    cv.imwrite(local_file, img)
    # formulate a path for the image on the S3 bucket
    img_loc = BUCKET + "images/" + local_file
    # formulate the S3 PUT command to push the image
    cmd = "s3cmd put --force " + local_file + " " + img_loc
    print("Executing command {}".format(cmd))
    subprocess.call(cmd, shell=True)

mqttclient = mqtt.Client()
mqttclient.on_connect = on_connect_local
mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
mqttclient.on_message = on_message

mqttclient.loop_start()

count=0
while True:
        time.sleep(20)
