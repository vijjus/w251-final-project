Converted the package notebook to a python script.

I ran this in a virtual environment since I started this before the container was up. I will test it later to see if it is missing any dependencies.

The script: Reads a video, crops faces, compares to known faces, prints if there is a match.

I noticed some things while doing this that we can discuss. Mainly how many frames for a faced to be matched in to consider it a good ID, and then how are we going to notify.
