# Connect to local MQTT Broker and subscribe to topic
# Convert message from bytes to image and save

import numpy as np
import cv2
import paho.mqtt.client as mqtt
import uuid

LOCAL_MQTT_HOST="mosquitto"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="system/jetson/webcam/face"

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
   print("Connected with result code "+str(rc))

   # Subscribing in on_connect() means that if we lose the connection and
   # reconnect then subscriptions will be renewed.
   client.subscribe(LOCAL_MQTT_TOPIC)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
   msg = msg.payload
   
   print("message recieved!")

   # Convert back from byte to array and then image
   face = np.frombuffer(msg, dtype=np.uint8)
   image = cv2.imdecode(face, flags=0)
   
   # Create a unique name for the image file
   name = "/mnt/mybucket/face_" + str(uuid.uuid4()) + ".png"

   # Write image to Obeject Storage
   cv2.imwrite(name, image)
   print("image written")

client = mqtt.Client("LocalClient-01")
client.on_connect = on_connect
client.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

client.on_message = on_message

# go into a loop to maintain connection
client.loop_forever()
