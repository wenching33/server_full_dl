import numpy as np
import cv2

def energy(A):
  eng = -999
  tmp = 0
  for i in range(len(A)):
    if A[i] == 0:
      tmp+=1
    else:
      if tmp>=eng:
        eng = tmp
      tmp=0
  if tmp>eng:
    eng = tmp
  return eng

def smooth(A,winSize = 5):
  sEng = np.zeros([len(A)])
  hw = int((winSize-1)/2)
  for i in range(len(A)):
    if i > hw and i < len(A)-hw:
      m = sum(A[i-hw:i+hw])/float(winSize)
      sEng[i] = m
    else:
      sEng[i] = 0
  return sEng

def trim(A,cut = False):
  m = sum(A)/float(len(A))
  A[A<m] = 0
  if cut:
    A[A>=m] -= m
  return A

def hasEmpty(np_blob):
  val = np_blob[:,1]
  m = sum(val)/float(val.shape[0])
  if len(val[val>m]) > val.shape[0]*0.6:
    return True
  else:
    return False

def getPeak(blob, width):
  pk = []
  np_blob = np.array(blob) #list to np array
  if np_blob.shape[0] > width*0.01 and hasEmpty(np_blob):
    #pk.append([np_blob[0,0],max(np_blob[:,1])])
    #pk.append([np_blob[-1,0],max(np_blob[:,1])])
    pk.append([int((np_blob[-1,0]+np_blob[0,0])/2.0),max(np_blob[:,1])])
  else:
    maxV = max(np_blob[:,1])
    tmp = np.where(np_blob[:,1]==maxV)
    ind = tmp[0][0]
    pk.append([np_blob[ind,0], max(np_blob[:,1])])
  return pk
  
def getPeaks(A):
  blobGap = 10
  leaveBlobCnt = 0
  blob = []
  blobStart = False
  peaks = []
  for i in range(len(A)):
    if A[i] != 0:
      if not blobStart:
        blobStart = True
      if blobStart:
        leaveBlobCnt = 0
        blob.append([i,A[i]])

    else:
      if blobStart:
        leaveBlobCnt+=1
        if leaveBlobCnt >= blobGap:
          blobStart = False
          leaveBlobCnt = 0
          pks = getPeak(blob,len(A))
          for pk in pks:
            peaks.append(pk)
          blob = []
        else:
          blob.append([i,A[i]])
    if i==len(A)-1 and blobStart:
      pks = getPeak(blob,len(A))
      for pk in pks:
        peaks.append(pk)
  return peaks


def autoSegment(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eg = cv2.Canny(gray,100,200) 
  egArr = np.zeros([img.shape[1]])
  for i in range(img.shape[1]):
    egArr[i] = energy(eg[:,i])
  egArr = smooth(egArr)
  trimedArr = []
  if len(egArr)>0:
    trimedArr = trim(egArr,False) # dont cut left value 
  if len(trimedArr)>0:
    trimedArr = trim(trimedArr,True) # cut left value with thv
  peaks = getPeaks(trimedArr)
  return peaks



