fp_g = open("report_gold.txt")
fp_in = open("report.txt")

golden_lines = fp_g.readlines()
input_lines = fp_in.readlines()
"""
fset = []
for line in input_lines:
  if line.startswith("./"):
    fset.append([input_lines.index(line),line])
"""    
def compare(inLine,gLine):
  inList = inLine[0:-2].split(",")
  #print("inList=")
  #print(inList)
  idset = []
  for item in inList:
    idset.append(item.split(":")[0])
  gList = gLine[0:-2].split(",")
  tpCnt = 0
  toDel = []
  for k in range(len(gList)):
    gitem = gList[k].split(":")
    if gitem[0] in idset:
      ind = idset.index(gitem[0])
      initem = inList[ind].split(":")
      #print("initem=")
      #print(initem)
      x_in = int(initem[1])
      y_in = int(initem[2])
      x_g = int(gitem[1])
      y_g = int(gitem[2])
      dist_thv = 60
      diff = pow(pow(x_g-x_in,2)+pow(y_g-y_in,2),0.5)
      #print("diff=%f"%diff)
      if(diff < dist_thv):
        #print("inside diff=%f"%diff)
        del inList[ind]
        del idset[ind]
        toDel.append(k)
        tpCnt+=1
      else:
        print("out diff=%f"%diff)
        continue
    else:
      continue
  toDel.reverse()
  #print("toDel=")
  #print(toDel)
  for e in toDel:
    del gList[e]
  msg = ""
  if len(gList)!=0:
    for e in gList:
      msg = msg+e+","
  fpCnt = len(inList)
  fnCnt = len(gList)
  acc = float(tpCnt)/(tpCnt+fpCnt+fnCnt)
  #print("acc=%f"%(acc))
  #print("tpCnt=%d, fpCnt=%d, fnCnt=%d"%(tpCnt,fpCnt,fnCnt))
  return acc,msg
    
    

accArr = []
for i in range(len(input_lines)):
  line = input_lines[i]
  if line.startswith("./"):
    if line in golden_lines:
      ind = golden_lines.index(line)
      acc,msg = compare(input_lines[i+1],golden_lines[ind+1])
      accArr.append([line.strip(),acc,msg])

#print(accArr)
s=0.0
for ele in accArr:
  print(ele[0]+", acc="+str(ele[1]))
  s+=ele[1]
  if ele[2] != "":
    print("Miss "+ele[2])
avgAcc = s/float(len(accArr))
print("Avg. accuracy=%f"%avgAcc) 
