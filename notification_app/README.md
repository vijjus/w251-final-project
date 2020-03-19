# w251-final-project # 

## Notification System ##

At the heart of IoT are web services that allow interaction with physical devices. “Smart” doorbells like Amazon’s Ring device have basic alert capabilities but lack more advanced analytic features such as the ability to differentiate known individuals from unknown individuals and adjust user alert notifications accordingly.
We propose adding the following baseline features to the Ring device to improve the user experience:

1)	a Deep Learning (DL) model that classifies the motion captured by the Ring device as a known person, unknown person or object; and learns the identities of people that the model and / or user determines are people of interest (.e.g., family member, babysitter, postal worker, etc.). 
2)	a notification system that transmits relevant information to the user via text message or email. 
In short, the model deciphers the signal (i.e., meaningful motion events) from the noise (non-meaningful motion events),  predicts whether the event warrants action,  then acts if action == warranted. 

### Research on Notification App ### 
Prior to creating the notification app, we researched other IoT security / surveillance projects to identify best practices. Three relevant and influential examples are shown below.

1.	“IoT Based Smart Home Security System with Alert and Door Access Control Using Smart Phone”
  - summary: 
     - custom security system that leverages a Raspberry Pi IoT device to capture images when visitors are detected at an entryway and transmits the image to the user as an email alert via TCP/IP. 
  - IoT Device Configuration: 
     - Raspberry Pi
  - Notification Framework: 
     - System -> Email via Simple Mail Transfer Protocol (SMTP) and Multi Purpose Internet Mail Extensions (MIME) 
     - Python Libraries: (1) mailutils  (2) ssmtp
  - Pseudo Code: 
     - step 1 -> define email address and message for alert
     - step 2 ->  define Rasberry Pi GPIO pins for PIR sensor 
     - step 3 -> define PiCamera function to capture image when PIR sensor triggers input
     - step 4 -> define Sendmail() module  
     
2.	“Home Security with Jetson Nano and Raspberry Pi 3” - https://github.com/dataplayer12/homesecurity
  - Summary: 
     - Raspberry Pi 3 camera performs motion detection and transmits recorded video to the Jetson Nano. Jetson Nano uses an object detection model to check if a person is identified in the video. If yes, the video recording and a screen shot of the person are sent by email. 
  - IoT Device Configuration: Raspberry Pi 3 camera + Jetson Nano  
  - Notification Framework:
    - System -> Email SMTP and MIME
    - Python Libraries: ssmtp
  - Pseudo Code: (See py script in directory)
  
3.	“How to Make a Smart Baby Monitor with Python, Rasberry Pi, Twillio SMS, and Peripheral Sensors”
   - Summary:  
     - The Raspberry Pi streams video over a local WiFi network, allowing any device on the network to watch the feed. The DS18B20 sensor takes a temperature reading every second, and sends that to the RPi device. The Python code reads that temperature data,  and informs the user via text notification when the temperature in the baby’s crib falls above or below a threshold.
   - IoT Device Configuration
     - Rasberry Pi + RPi Camera + DS18B20 Temperature Sensor
   - Notification Framework
     - System -> SMS via Twilio API
   - Python Libraries-> Twillio
   - Pseudo Code: (see py script in directory)
### Our Approach: TBD ###

Email and SMS notification systems both seem like viable options. I’ve created a Gmail account and Twilio account so that we can experiment with both. Username and password info is in the specific directory readme files. 
