import numpy as np
import cv2
vidcap = cv2.VideoCapture('IMG_0020.mov')
success,image = vidcap.read()
count = 0
while success:
    vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*50))
    cv2.imwrite("data/samples/frame%04d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
