import cv2
import utils

webcam = True
path = "phone_aruco_marker.jpg"
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

while True:
              if webcam:success,img = cap.read()
              else: img = cv2.imread(path)

              #img = cv2.resize(img,(1680,920))
              #cv2.imshow('Original output',img)
              img, conts = utils.getContours(img,showCanny=False,draw=True,minArea=2000,filter=4)

              if len(conts) != 0:
                            biggest = conts[0][2]
                            img_warp = utils.warpImg(img,biggest,630,840)
                            cv2.imshow('Warped',img_warp)

              cv2.imshow('Countored img',img)
              cv2.waitKey(1)