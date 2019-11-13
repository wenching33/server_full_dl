import cv2
import numpy as np
import os
def getOutputsNames(net):
  layersNames = net.getLayerNames()
  return [layersNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom, frame, classes):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, confThreshold, nmsThreshold, classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
      for detection in out:
        scores = detection[5:]
        classId = np.argmax(scores)
        confidence = scores[classId]
        if confidence > confThreshold:
          center_x = int(detection[0] * frameWidth)
          center_y = int(detection[1] * frameHeight)
          width = int(detection[2] * frameWidth)
          height = int(detection[3] * frameHeight)
          left = int(center_x - width / 2)
          top = int(center_y - height / 2)
          classIds.append(classId)
          confidences.append(float(confidence))
          boxes.append([left, top, width, height])
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    result = []
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
      i = i[0]
      box = boxes[i]
      left = box[0]
      top = box[1]
      width = box[2]
      height = box[3]
      drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame, classes)
      result.append([classIds[i],confidences[i],[int(left+0.5*width),int(top+0.5*height),width,height]])
    return result


def detect(imgPath, thresh, inpWidth, inpHeight, classesFile, modelConfig, modelWeights):
  frame = cv2.imread(imgPath)
  classes = None
  #confThreshold = 0.05  #Confidence threshold
  confThreshold = thresh
  nmsThreshold = 0.4   #Non-maximum suppression threshold

  with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
  net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
  net.setInput(blob)
  outs = net.forward(getOutputsNames(net))

  ret = postprocess(frame, outs, confThreshold, nmsThreshold, classes)
# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
  t, _ = net.getPerfProfile()
  label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
  cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
  e = imgPath.rfind('.')
  s = imgPath.rfind('/')
  outputFile = "./inferResult_cpu/"+imgPath[s+1:e]+"_out.JPG"
# Write the frame with the detection boxes
  cv2.imwrite(outputFile, frame.astype(np.uint8));
  return ret

if __name__=="__main__":
  classesFile = './dl/cfg/esl.names'
  classes = None
  with open(classesFile,'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
  
  modelConfig = './dl/cfg/yolov3_test.cfg'
  modelWeights = './dl/cfg/weights/yolov3_24240.weights'
  net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
  
  fp = open("input.txt","r")
  ind = 0
  for line in fp.readlines():
    imgPath = line.strip()
    frame = cv2.imread(imgPath)
    confThreshold = 0.01  #Confidence threshold
    nmsThreshold = 0.4   #Non-maximum suppression threshold
    inpWidth = 1824
    inpHeight = 1824

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs, confThreshold, nmsThreshold, classes)
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    output_folder_path = "./inferResult_cpu"
    outputFile = output_folder_path+"/out_"+str(ind)+".JPG"
    if not os.path.exists(output_folder_path):
      os.makedirs(output_folder_path)
    
    ind+=1
    cv2.imwrite(outputFile, frame.astype(np.uint8));

    #winName = "Deep learning"
    #cv2.imshow(winName, frame)
