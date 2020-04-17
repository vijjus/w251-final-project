#
# Monitor Directory and Run RingVideo on New Video Files
#

import os, time
path_to_watch = "./mnt/video_in"
before = dict ([(f, None) for f in os.listdir (path_to_watch)])
while 1:
  time.sleep (10)
  after = dict ([(f, None) for f in os.listdir (path_to_watch)])
  added = [f for f in after if not f in before]
  filename = str(added).strip("'[]'")
  new_filename = "processed_" + filename
  command = "python RingVideo.py " + filename + " " + new_filename
  if added: os.system(command)
  before = after
