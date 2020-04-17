### Information on running the python script RingVideo.py

The Container was run in an IBM Cloud VM with Oject storage buckets mounted to the /mnt/ folder

#### Run with the following information

Start the docker contianer with the following command:
    docker run --name ring -v /mnt:/mnt -it ring bash  

Then start the Watchdog script with the following:
    python RingWatchdog.py

This will monitor the /mnt/video_in/ folder for new videos. Once processed the new file will be moved to the /mnt/video_out/ folder 
