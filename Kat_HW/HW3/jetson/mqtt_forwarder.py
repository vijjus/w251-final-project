# MQTT message forwarder
# Connect to local and remote brokers
# subscribe to to topic and forward messages

import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST="mosquitto"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="system/jetson/webcam/face"

REMOTE_MQTT_HOST="169.62.93.58"
REMOTE_MQTT_PORT=1883
REMOTE_MQTT_TOPIC="system/jetson/webcam/face"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)

def on_connect_remote(client, userdata, flags, rc):
        print("connected to remote broker with rc: " + str(rc))
        client.subscribe(REMOTE_MQTT_TOPIC)
	
def on_message(client,userdata, msg):
  try:
    print("message received!")	
    # re-publish the message
    msg = msg.payload
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
  except:
    print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client("LocalClient-01")
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

remote_mqttclient = mqtt.Client("RemoteClient-01")
remote_mqttclient.on_connect = on_connect_remote
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)

local_mqttclient.on_message = on_message

# go into a loop to maintain connection
local_mqttclient.loop_forever()
remote_mqttclient.loop_forever()
