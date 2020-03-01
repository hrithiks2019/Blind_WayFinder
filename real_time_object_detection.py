# Libraries Import Panikirom:
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import pyttsx3
# Text to Speech module ah Initialize pandrom:
engine = pyttsx3.init()

kp = ['prototxt.txt','trainedmodel.caffemodel','blind','wayfinder','detection','object','hrithik']
# classifier la iruka objects oda array
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# ovovoru object um different set of colour code pandrom:
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading prototxt file...")
print("[INFO] loading model...")

# classifier ah initialize panikirom:
net = cv2.dnn.readNetFromCaffe(kp[0], kp[1])
print("[INFO] Getting Video stream From Webcam for Object detection...")

# camera la irunthu video input yeduthukirom:
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# main loop:
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()

	# object mela box bound pandrathuku
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.2:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label = "{}: {:.2f}%".format(CLASSES[idx],confidence * 100)
		
			# classify pana object oda name ah soldrathuku:
			if confidence*100 > 80:
				print(f'{CLASSES[idx]} is infront of you')
				engine.say(str(CLASSES[idx]))
				print(str(CLASSES[idx]))
			
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			
				

	cv2.imshow(str(kp[-1]+kp[-2]+kp[-3]), frame)
	key = cv2.waitKey(1) & 0xFF
	
	# q button press panuna program close aaidum:
	if key == ord("q"):
		break
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
