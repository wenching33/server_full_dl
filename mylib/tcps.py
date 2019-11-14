#import json
import simplejson as json
import socket,traceback
import numpy as np
import threading
import time
from main import *
import cv2 
import struct
import unicodedata
import darknet_modi as dnet
index=0
id_list = []
name_list = []
price_list = []

def initProductInfo():
  path = '/home/wenching/ESL/server_full_dl/train_set'
  f=open(path,'r')
  for imagePath in f.readlines():
    imagePath = imagePath.strip()
    print(imagePath)
    tmp = imagePath.split(",")
    id_list.append(tmp[1]) #product id
    name_list.append(tmp[2]) #product name
    price_list.append(tmp[3]) #product price

netMain, metaMain = dnet.prepareDetector(configPath = "./dl/cfg/yolov3_test.cfg", weightPath = "./dl/cfg/weights/yolov3_24240.weights", metaPath = "./dl/cfg/esl.data", showImage = True, makeImageOnly = False, initOnly = False)

initProductInfo()
"""
host='192.168.43.97'
port=33333
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind((host,port))
s.listen(1)
"""
while 1:
  try:
    """
    print("Waiting for connect")
    clientsock,clientaddr=s.accept()
    print("connection comes from")
    print(clientsock.getpeername())
    #datain = clientsock.recv(4)
    size = clientsock.recv(4)
    barr = bytearray(size)
    size_int = struct.unpack('>i',barr)[0]
    f=open("myPic.JPG",'wb+')
    byteNumRecv = 0
    begin_t =time.time()
    while 1:
      datain = clientsock.recv(999999999)
      barr = bytearray(datain)
      f.write(barr)
      byteNumRecv += len(barr)
      if byteNumRecv == size_int:
        break
    f.close()
    end_t = time.time()
    print("image received. Cost %f seconds"%(float(end_t-begin_t)))
    test_img_path = "myPic.JPG"
    """
    test_img_path = "../myPic.JPG" #ok 15,16 both merge; find edge ok
    begin_t_0 = time.time()
    ipinf, cropInfo, inf = getInfo(test_img_path, netMain, metaMain)
    end_t_0 = time.time()
    print("Time cost = %f seconds"%(end_t_0-begin_t_0))
        
    #check
    if len(ipinf) != len(inf):
      print("Error! Number of ESL IP is not equal to number of products line.")
         
    products = []
    for pl in range(len(inf)):
      ip = unicodedata.normalize('NFKD',ipinf[pl][0]).encode('ascii','ignore')
      port = unicodedata.normalize('NFKD',ipinf[pl][1]).encode('ascii','ignore')
      X = cropInfo[pl][0] 
      Y = cropInfo[pl][1]
      Width = cropInfo[pl][2]
      Height = cropInfo[pl][3]
      info_list = []
      for p in inf[pl]:
        # p[0]: product id, p[1]: x position wrt left green marker
        print("product %d at %d in layer %d"%(p[0],p[1],pl+1))
        id = id_list[p[0]-1]
        name = name_list[p[0]-1]
        price = price_list[p[0]-1]
        xx = p[1]
        info_dict = {"id":id,"name":name,"price":price,"x":xx}
        info_list.append(info_dict)
      ret_dict = {"ip":ip,"port":port,"x":X,"y":Y,"width":Width,"height":Height,"info":info_list}
      products.append(ret_dict)
    ret_str = json.dumps(products)
    ret_str = ret_str.encode('utf-8')
    sent_len = len(ret_str)
    tmp = struct.pack('>i',sent_len)
    #tmp = tmp.decode('utf-8','ignore')
    #break
    #clientsock.sendall(tmp)
    #clientsock.sendall(ret_str)
    #clientsock.close()
    break
  
  except KeyboardInterrupt:
    raise
  except:
    traceback.print_exc()
    continue
