import cv2
import numpy as np

def getContours(img,cThr=[100,100],showCanny=False,minArea=1000,filter=0,draw=False):
              img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
              img_blur = cv2.GaussianBlur(img_gray,(5,5),1)
              img_canny = cv2.Canny(img_blur,cThr[0],cThr[1])

              kernel = np.ones((5,5))
              img_dial = cv2.dilate(img_canny,kernel,iterations=4)
              img_thresh = cv2.erode(img_dial,kernel,iterations=2)
              if showCanny: cv2.imshow('Canny',img_thresh)

              contours,hierarchy = cv2.findContours(img_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

              final_contours = []
              for i in contours:
                            area = cv2.contourArea(i)

                            if area > minArea:
                                          peri = cv2.arcLength(i,True)
                                          approx = cv2.approxPolyDP(i,0.02*peri,True)
                                          bbox = cv2.boundingRect(approx)

                                          if filter > 0:
                                                        if len(approx) == filter:
                                                                      final_contours.append((len(approx),area,approx,bbox,i))
                                          else:        
                                                        final_contours.append((len(approx),area,approx,bbox,i))

              final_contours = sorted(final_contours,key=lambda x: x[1],reverse=True)

              if draw:
                            for con in final_contours:
                                          cv2.drawContours(img,con[4],-3,(255,255,255),3)            

              return img, final_contours 

def reorder(points):
              points_new = np.zeros_like(points)
              points = points.reshape((4,2))
              add = np.sum(1)
              points_new[0] = points[np.argmin(add)]
              points_new[3] = points[np.argmax(add)]
              diff = np.diff(points,axis=1)
              points_new[1] = points[np.argmin(diff)]
              points_new[2] = points[np.argmax(diff)]

              return points_new


def warpImg(img,points,w,h):
              points = reorder(points)
              pts1 = np.float32(points)
              pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
              matrix = cv2.getPerspectiveTransform(pts1,pts2)
              img_warp = cv2.warpPerspective(img,matrix,(w,h))
              return img_warp