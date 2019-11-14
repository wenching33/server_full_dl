from pyzbar import pyzbar
import threading
from util import *
#from matplotlib import pyplot as plt
import cv2
from autoSegment import *
from markerFinder import *
import time
import opencvDarknet as dnet


def getInfo(test_img_path, size):
  ###### Get marker position & crop barcode #####
  test_img = cv2.imread(test_img_path)
  begin_t = time.time()
  n,pts = findMarker(test_img, ratio=1.0)
  end_t = time.time()
  print("Find markers cost: %f seconds.\n"%(end_t-begin_t))
  marker_y = []
  ipinf = []
  cropInfo = []
  begin_t = time.time()
  ind_cp = 0
  for j in range(0,n,2):
    y = int((pts[j][1]+pts[j+1][1])/2.0)
    marker_y.append(y)
    xLeft = int(min(pts[j][0],pts[j+1][0]))
    xRight = int(max(pts[j][0],pts[j+1][0]))
    W = xRight-xLeft
    if j == 0:
      H = marker_y[j]
    else:
      H = marker_y[int(j/2)] - marker_y[int(j/2-1)]
    cropInfo.append([xLeft,y,W,H])
    print("xLeft=%d, y=%d, W=%d, H=%d"%(xLeft,y,W,H))
    cpimg = test_img[y-180:y+180,xLeft-40:xRight+40]
    fn="check_barcode_"+str(ind_cp)+".JPG"
    t = cv2.cvtColor(cpimg,cv2.COLOR_BGR2GRAY)
    rr, t = cv2.threshold(t,127,255,cv2.THRESH_BINARY)
    #t = cpimg
    cv2.imwrite(fn,t)
    ind_cp+=1
    barcodes = pyzbar.decode(t)
    barcodes = []
    if len(barcodes)==0:
      ipinf.append(["-1","-1"])
    else:
      for barcode in barcodes:
        # our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        tmp = barcodeData.split(',')
        ip = tmp[0]
        ip = str(int(ip[0:3]))+"."+str(int(ip[3:6]))+"."+str(int(ip[6:9]))+"."+str(int(ip[9:12]))
        port = "0"
        ipinf.append([ip,port])
  marker_y = sorted(marker_y,key = lambda x:x)
  end_t = time.time()
  print("Crop ,detect barcode, and get cropInfo cost: %f seconds.\n"%(end_t-begin_t)) 
  
  ##### Detect product and push them into each row of product #####
  products = [[] for i in range(len(marker_y))]
  before = time.time()
  thresh = 0.01
  inpW = size
  inpH = size
  detections = dnet.detect(test_img_path, thresh, inpW, inpH, "./dl/cfg/esl.names", "./dl/cfg/yolov3_test.cfg", "./dl/cfg/weights/yolov3_24240.weights")
  #print(detections) 
  products_list = []
  for i in range(len(detections)):
    cls = detections[i][0]+1
    x = int(detections[i][2][0])
    y = int(detections[i][2][1])
    conf = detections[i][1]
    products_list.append([cls,x,y,conf])
  for k in range(len(products_list)):
    for n_ in range(len(marker_y)):
      if products_list[k][2] < marker_y[n_]:
        products[n_].append(products_list[k])
        break
  after = time.time()
  print("Detect 20 products cost: %f seconds.\n"%(after-before))
  print(products)
  finalPos = []
  begin_t = time.time()
  for i in range(len(marker_y)):
    p = products[i]
    p = sorted(p, key=lambda x:x[1])
    p = checkDist(p)
    x_min = min(int(pts[i*2][0]), int(pts[i*2+1][0]))
    offset = max(x_min-10,0)
    fPos = []
    for pi in (p):
      fPos.append([pi[0],pi[1]-offset,pi[2]])  #pi[k] is a product, p[k] = [cls,x,y,confi.]
    finalPos.append(fPos)
    end_t = time.time()
    print("Find positions of products line_%d cost %f seconds.\n"%(i,end_t-begin_t))
    for ps in fPos:
      cv2.rectangle(test_img,(int(ps[1]-10)+offset,int(marker_y[i]-10)),(int(ps[1]+10)+offset,int(marker_y[i]+10)),(0,0,255),2)
      #print("class:%d, x:%d"%(pd[0],pd[1]))
      cv2.putText(test_img, str(ps[0]), (int(ps[1])+offset, int(marker_y[i])), cv2.FONT_HERSHEY_DUPLEX,3, (0, 255, 255), 3, cv2.LINE_AA)
    
  #show for check
  cv2.imwrite("checkResult.JPG",test_img)
  #resizedImg = cv2.resize(test_img, (int(0.2*test_img.shape[1]), int(0.2*test_img.shape[0])), interpolation=cv2.INTER_CUBIC)
  #cv2.imshow("final",resizedImg)
  #cv2.waitKey(0)
    
  return ipinf, cropInfo, finalPos


