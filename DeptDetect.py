import numpy as np
import cv2
from matplotlib import pyplot as plt
 
camera_port_0 = 0
camera_port_1 = 1
 
ramp_frames = 30
 
camera_0 = cv2.VideoCapture(camera_port_0)
camera_1 = cv2.VideoCapture(camera_port_1)
 
camera_0.set( cv2.cv.CV_CAP_PROP_MODE, cv2.cv.CV_8UC3 );
camera_1.set( cv2.cv.CV_CAP_PROP_MODE, cv2.cv.CV_8UC3 );

for i in xrange(ramp_frames):
    retval, im0 = camera_0.read()
    retval, im0 = camera_1.read()
print("Taking image...")

while(True):
    retval, imgL = camera_0.read()
    retval, imgR = camera_1.read()

    frameL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    frameR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    window_size = 5
    min_disp = 32
    num_disp = 112-min_disp

    ##stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET, numDisparities=16, blockSize=15)
    stereo = cv2.StereoSGBM(
        minDisparity = min_disp,
        numDisparities = num_disp,
        SADWindowSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 1,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        fullDP = False
    )
    disparity = stereo.compute(frameL,frameR).astype(np.float32) / 16.0
    #disparity = stereo.compute(image_left, image_right).astype(np.float32) / 16.0
    disparity = (disparity-min_disp)/num_disp

    
    cv2.imshow("TestR",frameR)
    cv2.imshow("TestL",frameL)
    cv2.imshow("Test",disparity)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

del(camera_0)
del(camera_1)
