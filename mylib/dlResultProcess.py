
def getCls(name):
  fp = open('./dl/cfg/esl.names','r')
  menu = fp.readlines()
  return menu.index(name+'\n')
  

