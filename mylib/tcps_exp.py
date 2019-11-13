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
import argparse
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


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str, default='../myPic.JPG', help='path to img')
  args = parser.parse_args()
  test_img_path = args.path

  netMain, metaMain = dnet.prepareDetector(configPath = "./dl/cfg/yolov3_test.cfg", weightPath = "./dl/cfg/weights/yolov3_24240.weights", metaPath = "./dl/cfg/esl.data", showImage = True, makeImageOnly = False, initOnly = False)

  initProductInfo()
  fp = open("input.txt")
  fp_report = open("report.txt","w")

  for line in fp.readlines():
    test_img_path = line.strip()
    fp_report.write(test_img_path+"\n")
    try:
      #test_img_path = "../myPic.JPG" #ok 15,16 both merge; find edge ok
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
        #fp_report.write("Row")
        for p in inf[pl]:
          # p[0]: product id, p[1]: x position wrt left green marker
          print("product %d at %d in layer %d"%(p[0],p[1],pl+1))
          id = id_list[p[0]-1]
          name = name_list[p[0]-1]
          price = price_list[p[0]-1]
          xx = p[1]
          info_dict = {"id":id,"name":name,"price":price,"x":xx}
          info_list.append(info_dict)
          fp_report.write("%d:%d:%d,"%(p[0],p[1],p[2]))
        ret_dict = {"ip":ip,"port":port,"x":X,"y":Y,"width":Width,"height":Height,"info":info_list}

        products.append(ret_dict)
      fp_report.write("\n")
      ret_str = json.dumps(products)
      ret_str = ret_str.encode('utf-8')
      sent_len = len(ret_str)
      tmp = struct.pack('>i',sent_len)
      #tmp = tmp.decode('utf-8','ignore')
      #break
      #clientsock.sendall(tmp)
      #clientsock.sendall(ret_str)
      #clientsock.close()
      #break
    except KeyboardInterrupt:
      raise
    except:
      traceback.print_exc()
      continue
  fp_report.close()
