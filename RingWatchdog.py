#
# Monitor Directory and Run RingVideo on New Video Files
#

import os, time
path_to_watch = "."
before = dict ([(f, None) for f in os.listdir (path_to_watch)])
while 1:
  time.sleep (10)
  after = dict ([(f, None) for f in os.listdir (path_to_watch)])
  added = [f for f in after if not f in before]
  filename = str(added).strip("'[]'")
  new_filename = "processed_" + filename
#  removed = [f for f in before if not f in after]
#  command = ("python RingVideo.py {} {}".format(added, new_file))
  command = "python RingVideo.py " + filename + " " + new_filename
  # python RingVideo.py RingVideo_3.mp4 FamilyID_3.avi
#  if added: print("Added: ", ", ".join (added))
  if added: os.system(command)
#  if removed: print "Removed: ", ", ".join (removed)
  before = after
