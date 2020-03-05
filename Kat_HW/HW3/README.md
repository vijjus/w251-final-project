# Homework 3 - Katayoun Borojerdi
The objective of this homework was to build an IoT like application pipeline with components running both on the edge (my Nvidia Jetson TX2) and the cloud (VM created using IBM cloud). I used docker and docker-compose to conatinerize the applications which would make it quick and easy to deploy and run in a real appliation. Contianers are based in either Alpine of Ubuntu depending on the services they run. The applications use MQTT to communicate, with the applications written in Python.

The pipeline capture faces in a video stream coming from a webcam connected to the Jetson in real time, transmit them to the cloud, and saves the faces in an object sotorage in the cloud.

## Cloud Application - IBM Cloud Virtual Machine
In the cloud, I started by creating a lightweight virtual machine (1 CPUs and 2 G of RAM). Then install docker and dokcer-compose. I created the Object Storage and mounted it. Then I could run an MQTT broker so that my jetson can connect to the cloud VM. The incoming faces are sent as binary messages so the service that subscribes to the broker locally recieves them and converts them back into image files, and then saves them to the cloud Object storage.

### Provision VM
I setup up a lightweight VM from my jumpbox using the CLI 

The following command creates the new virutal machine:
```
ibmcloud sl vs create --hostname=faces --domain=storage.cloud --cpu=1 --memory=2048 --datacenter=sjc04 --os=UBUNTU_18_64 --san --disk=100 --key=1689110
```
Once the vm is up and running I find the ip address to ssh in using ``` ibmcloud sl vs list ``` and then I followed the instructions from hw2 in order to harden the VSI and ensure the password login is disabled.  

### Install Docker and Docker-Compose
Next I install docker and docker-compose to run my containers on this VM.

I followed the instruction from week2 lab2 to install Docker and verify that it is working properly
```
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
	
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"	

apt-get update

apt-get install -y docker-ce
```
verify
```
docker run hello-world
```

Next to install Docker Compose I ran the following

```
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
```

### Setup IBM Object Storage

I created the Object Storage through IBM Cloud service website. https://cloud.ibm.com
I did this by going to resource list menu and clicking on the "create resource" blue button on the top right. 
Next, by clicking the storage menu I selected "Object Storage" and select the default "Lite Plan".

Next I mounted the cloud object VSI using the following
```
#Biuld and install package
sudo apt-get update
sudo apt-get install automake autotools-dev g++ git libcurl4-openssl-dev libfuse-dev libssl-dev libxml2-dev make pkg-config
git clone https://github.com/s3fs-fuse/s3fs-fuse.git
Add storage

#Build and install library
cd s3fs-fuse
./autogen.sh
./configure
make
sudo make install
```
Next I went to my object storage and from the service credentials viewed my credentials:

```
   "cos_hmac_keys": {
    "access_key_id": "somekey",
    "secret_access_key": "somesecretkey"
  },
```

And substitued this values for my <Access_Key_ID> and <Secret_Access_Key> in the command below.
```
echo "<Access_Key_ID>:<Secret_Access_Key>" > $HOME/.cos_creds
chmod 600 $HOME/.cos_creds
```

Finally I created a directory to mount my bucket using the following
```
sudo mkdir -m 777 /mnt/mybucket
sudo s3fs bucketname /mnt/mybucket -o passwd_file=$HOME/.cos_creds -o sigv2 -o use_path_request_style -o url=https://s3.us-east.objectstorage.softlayer.net
```

### Starting the MQTT Broker and Image Processor/Saver on the Cloud VSI
I started with the Mosquitto Broker running on Alipne Linux. I created a Dockerfile.alpine-mqtt and built the docker image with the following command.
```
docker build -t mqtt-broker -f Dockerfile.mqtt-broker .
```
In order to get the incoming images from the MQTT broker and and save them to the Object Storage. I used a python file to subscribe to the broker and convert the mesages back to an image from bytes and save it to the object storage. I created a Dockerfile based on Ubuntu and added the neccesary apps and build the docker image.
```
docker build -t image-saver -f Dockerfile.opencv-mqtt-image-save .
```

To make it easy to run all containers I used a docker-compose.yml file to spin up both services at once with the follwoing command. This file starts the mosquitto broker and also runs the python image_saver. 
```
docker-compose up
```

## Edge Application - Jetson TX2
For the edge face detector component, I used OpenCV writen in Python to stream video frames coming from a connected USB camera. When one or more faces are detected in the frame, the application cuts them out of the frame and send via a binary message to the MQTT broker. The application shows the frame in the stream and also shows the image in a new window for verification. There is also a MQTT forwarder running that receives these messages from the local broker, and sends them to the MQTT broker running on the IBM Cloud VM.

### Starting the MQTT Broker, Forwarder and Image Capture on the jetson TX2
The same dockerfile use to build the broker on the VM can be re-used on the Jetson. Also the Processor/Saver dockerfile can be used to build the container for face detection. I only had to swap out the python file that subscribes and saves the image, to the file that turns on video from the webcam and detects faces. For the Forwader I found a alpine linux image that included python, and used that as my base image, then added Mosquitto, paho-mqtt and a python file. The python file subscribes to both the local Jetson broker and the remote VM broker. The file then sucribes to a local topic and forwards all messages to the remote broker. 

Docker was installed on the Jetson in a previous excercise. First I created the neccesary images for each service.  

To build the MQTT Broker on the Jestson I used the following:
```
docker build -t mqtt-broker -f Dockerfile.mqtt-broker .
```

To build the MQTT Forwarder on the Jestson I used the following command:
```
docker build -t mqtt-forwarder -f Dockerfile.mqtt-forwarder .
```

To build the container that turns on the video capture and runs face detection:
```
docker build -t face-detect -f Dockerfile.opencv-mqtt-face-detect .
```
Again to make it easy to run all containers I used a docker-compose.yml file to spin up all three services at once with the follwoing command. I did not start the face detection script to make it easeier to start and stop.

I installed docker-compose on the jetson. Following the guide [here](https://blog.hypriot.com/post/nvidia-jetson-nano-install-docker-compose/)
TLDR run the following:
```
# step 1, install Python PIP
sudo apt-get update -y
sudo apt-get install -y curl
curl -SSL https://bootstrap.pypa.io/get-pip.py | sudo python

# step 2, install Docker Compose build dependencies for Ubuntu 18.04 on aarch64
sudo apt-get install -y libffi-dev
sudo apt-get install -y python-openssl

# step 3, install latest Docker Compose via pip
export DOCKER_COMPOSE_VERSION=1.25.1
sudo pip install docker-compose=="${DOCKER_COMPOSE_VERSION}"
```

Then run the docker-compose file
```
docker-compose up
```

To start the video and face detection I used the following command:
```
# need to run xhost only once
xhost + 
docker-compose exec face-detect python video_face_detect.py
```

Now that all the services are running and the video is streaming to a window on the Jetson, if a face is detected it will capture and save the image. Here is a [link to sample face image](https://s3.us-south.cloud-object-storage.appdomain.cloud/cloud-object-storage-w251-hw3-faces/face_04e447f4-318f-4686-9446-97ac9bcdde6b.png) and many other images are saved in the base folder.

I chose the naming for the topic as "system/jetson/webcam/face". The reason for this is, I created a base container for an entire IoT system, another layer for my jetson, a layer for the webcam input and finally the face topic. I did this so that i the future if I wanted to detect other items from the webcam I could do that in anthor topic. Or if I hooked up other sensors I would be able to create a seperate topic. And so on if there were other edge devices or sensors contected. I used QoS 0 as it seemed like there were many frames that were deteced repeatedly, and if a single image was lost for some reason I did not think it would make much difference in this application.
