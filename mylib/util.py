import numpy as np
def median(A):
  if len(A)==0:
    check=1
  return A[len(A)/2]
def removeOutlier_simple(A):
  lastLenA = len(A)
  while 1:
    if lastLenA < 1:
      return []
    if lastLenA == 1:
      return A
    A = np.array(sorted(A,key = lambda x:x[0]))
    X = A[:,0]
    Q1_x = median(X[:len(X)/2])
    Q3_x = median(X[len(X)/2:])
    IQR_x = abs(Q3_x-Q1_x)
    k_x = 1.8
    toRmX = []
    for i in range(len(X)):
      if X[i] < Q1_x-k_x*IQR_x or X[i] > Q3_x+k_x*IQR_x:
        toRmX.append(i)
    A = np.delete(A,toRmX,0)
    
    A = np.array(sorted(A,key = lambda x:x[1]))
    Y = A[:,1]
    Q1_y = median(Y[:len(Y)/2])
    Q3_y = median(Y[len(Y)/2:])
    IQR_y = abs(Q3_y-Q1_y)
    k_y = 1.8
    toRmY = []
    for i in range(len(Y)):
      if Y[i] < Q1_y-k_y*IQR_y or Y[i] > Q3_y+k_y*IQR_y:
        toRmY.append(i)
    A = np.delete(A,toRmY,0)
    
    if len(A)==lastLenA:
      break
    lastLenA = len(A)
  return A
    
  
def checkDist(p):
  if len(p)<=1:
    return p
  while(1):
    p_ = list(p)
    for i in range(1,len(p)):
      deltaX = abs(p[i][1]-p[i-1][1])
      if deltaX < 120:
        if p[i][3] < p[i-1][3]:
          p_.remove(p[i])
        else:
          if p[i-1] in p_:
            p_.remove(p[i-1])
    if len(p_) == len(p):
      break
    else:
      p = p_

  return p_

def mean(A):
  return sum(A)/float(len(A))

def max_deviation(A):
  mu = mean(A)
  dev = [(a, abs(a-mu)) for a in A]
  dev.sort(key=lambda k: k[1], reverse=True)
  return dev[0][0]

def remove_outliers(A, tol=.1):
  if len(A) == 0:
    return A 
  mu = mean(A)
  out = max_deviation(A)
  A.remove(out)
  if len(A) == 0:
    return A 
  mu_prime = mean(A)
  if abs(mu_prime - mu)/float(mu) > tol:
    return remove_outliers(A, tol)
  else:
    A.append(out)
    return A

def removeOutlier(pos):
  X = list(pos[:,0])
  Y = list(pos[:,1])
  ans = [[x,y] for x,y in pos if x in remove_outliers(X) and y in remove_outliers(Y)]
  return np.array(ans)
