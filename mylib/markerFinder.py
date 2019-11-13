import cv2
import numpy as np
import matplotlib.pyplot as plt

def findMarker(img, ratio):
  #img = cv2.resize(img, (int(ratio*img.shape[1]), int(ratio*img.shape[0])), interpolation=cv2.INTER_CUBIC)
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv, (40,120, 190), (80, 255, 255))
  # slice the green
  imask = mask>0
  selected = np.zeros_like(img, np.uint8)
  selected[imask] = img[imask]
  #cv2.imwrite("r1.JPG",selected)

  kernel = np.ones((15, 15), np.uint8)
  #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
   
  eroded = cv2.erode(selected, kernel, iterations=1)
  dilated = cv2.dilate(eroded, kernel, iterations=1)
  ret,dilated = cv2.threshold(dilated,127,255,cv2.THRESH_BINARY_INV)
  #dilated = cv2.cvtColor(dilated, cv2.COLOR_HSV2GRAY)
  cv2.imwrite("checkDilated.JPG",dilated)
  h,s,v = cv2.split(dilated)
  dilated = s #get gray channel
  cv2.imwrite("checkGray.JPG",dilated)
  
  #resizedImg = cv2.resize(img, (int(dilated.shape[1]), int(dilated.shape[0])), interpolation=cv2.INTER_CUBIC)

  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()
  params.filterByColor = False
  #params.blobColor = 255
  
  #Change thresholds
  params.minThreshold = 10
  params.maxThreshold = 200
  
  # Filter by Area.
  params.filterByArea = False
  params.minArea = 100
   
  # Filter by Circularity
  params.filterByCircularity = False
  #params.minCircularity = 0.1
    
  # Filter by Convexity
  params.filterByConvexity = False
  #params.minConvexity = 0.87
     
  # Filter by Inertia
  params.filterByInertia = False
  #params.minInertiaRatio = 0.01
  det = cv2.SimpleBlobDetector_create(params)
  #det = cv2.SimpleBlobDetector_create()
  showRatio = 0.5
  """
  keypoints = det.detect(dilated)
  dilated = cv2.drawKeypoints(dilated, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  showImg = cv2.resize(dilated, (int(showRatio*dilated.shape[1]), int(showRatio*dilated.shape[0])), interpolation=cv2.INTER_CUBIC)
  plt.imshow(showImg),plt.show()
  """
  dilated = cv2.bitwise_not(dilated)
  keypoints = det.detect(dilated)
  check = cv2.drawKeypoints(dilated, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  cv2.imwrite("checkMarkerPos.JPG",check)
  #showImg = cv2.resize(dilated, (int(showRatio*dilated.shape[1]), int(showRatio*dilated.shape[0])), interpolation=cv2.INTER_CUBIC)
  #plt.imshow(showImg),plt.show()
  print("Blob number = %d"% len(keypoints))
    
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#dh = dilated.shape[0]
#dw = dilated.shape[1]
#dilated = cv2.drawKeypoints(dilated, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#showImg = cv2.resize(dilated, (int(0.2*w), int(0.2*h)), interpolation=cv2.INTER_CUBIC)
#cv2.imshow("test",showImg)
#cv2.waitKey(0)
  pts = []
  for i in range(len(keypoints)):
    pt = [keypoints[i].pt[0],keypoints[i].pt[1],keypoints[i].size]
    pts.append(pt)
  pts = sorted(pts,key=lambda x:(x[1],x[0]))
  """
  if len(pts)<2 or len(pts)%2 !=0:
    if len(pts) < 2:
      print("blob detection fail")
      return 0,[]
    else:
      check=1
  """
  #for i in range(0,len(pts),2):
  #  cv2.rectangle(resizedImg,(int(pts[i][0]-10),int(pts[i][1]-10)),
  #       (int(pts[i+1][0]+10),int(pts[i+1][1]+10)),(0,0,255),2)
  #showImg = cv2.resize(resizedImg, 
  #			 (int(0.4*resizedImg.shape[1]),int(0.4*resizedImg.shape[0])), 
  #			 interpolation=cv2.INTER_CUBIC)
  return len(pts),pts


